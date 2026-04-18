"""Build drift_prev (and optional pixel drift u,v) from several earlier GeoTIFF scenes.

Uses:
- Sentinel-1 style datetimes embedded in filenames (YYYYMMDDTHHMMSS).
- Geolocation from each GeoTIFF (CRS, transform, bounds).
- Spatial overlap in WGS84 to pick valid predecessors.
- rasterio.reproject to align previous rasters onto the current grid.
- Temporal exponential weights over multiple previous acquisitions.
- Dense pixel displacement (prev_blend -> current) via OpenCV Farneback optical flow
  when available; otherwise phase correlation on downscaled images for a constant
  shift applied as a crude flow field.

Typical usage (folder of full-scene IceClass GeoTIFFs):

    python generate_drift_from_history.py ^
        --scenes-dir "Dataset_2025_IceClass" ^
        --output-dir "drift_from_history" ^
        --num-prev 4 ^
        --min-overlap 0.02 ^
        --num-classes 9

Outputs per reference scene (same basename as input):
    <output-dir>/<stem>_drift_prev.tif   — uint8, same shape/CRS as reference
    <output-dir>/<stem>_drift_uv.tif   — 2 bands float32: u,v in **pixels** on ref grid
    <output-dir>/<stem>_drift_meta.json — which predecessors were used, times, weights

This is meant for business / training pipelines where real temporal stacks exist.
It does not modify your existing prepare.py synthetic drift.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds


# ---------------------------------------------------------------------------
# Filename time parsing (Sentinel-1 product id fragment)
# ---------------------------------------------------------------------------


def parse_s1_acquisition_times(stem: str) -> Tuple[datetime, Optional[datetime]]:
    """Parse first (and optional second) YYYYMMDDTHHMMSS from filename stem."""
    matches = re.findall(r"(\d{8}T\d{6})", stem)
    if not matches:
        raise ValueError(f"No S1-style datetime (YYYYMMDDTHHMMSS) found in: {stem}")
    start = datetime.strptime(matches[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    end = None
    if len(matches) >= 2:
        end = datetime.strptime(matches[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return start, end


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def bounds_to_wgs84(bounds: Tuple[float, float, float, float], crs: Any) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return transform_bounds(crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)


def geo_bounds_iou(
    a_bounds: Tuple[float, float, float, float],
    a_crs: Any,
    b_bounds: Tuple[float, float, float, float],
    b_crs: Any,
) -> float:
    """Approximate IoU of geographic footprints in lon/lat (degrees)."""
    al, ab, ar, at = bounds_to_wgs84(a_bounds, a_crs)
    bl, bb, br, bt = bounds_to_wgs84(b_bounds, b_crs)
    ix0 = max(al, bl)
    iy0 = max(ab, bb)
    ix1 = min(ar, br)
    iy1 = min(at, bt)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    aw = max(0.0, ar - al) * max(0.0, at - ab)
    bw = max(0.0, br - bl) * max(0.0, bt - bb)
    union = aw + bw - inter
    return float(inter / union) if union > 0 else 0.0


def hours_between(t_ref: datetime, t_prev: datetime) -> float:
    return (t_ref - t_prev).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Reprojection & encoding
# ---------------------------------------------------------------------------


def reproject_band_to_reference(
    src_path: Path,
    ref_transform: rasterio.Affine,
    ref_crs: Any,
    ref_height: int,
    ref_width: int,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Warp one band of src onto the reference grid (nearest)."""
    with rasterio.open(src_path) as src:
        dst = np.zeros((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata if src.nodata is not None else nodata,
            dst_nodata=np.nan,
        )
    return dst


def class_codes(num_classes: int) -> np.ndarray:
    return np.rint(np.linspace(1, 255, num_classes)).astype(np.float32)


def snap_to_codebook(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Snap each pixel value to nearest class code (vectorized)."""
    v = values.astype(np.float32)
    # (H,W,1) - (1,1,K) -> argmin over K
    d = np.abs(v[..., None] - codes.reshape(1, 1, -1))
    idx = np.argmin(d, axis=-1)
    return codes[idx].astype(np.uint8)


def temporal_blend(
    layers: List[np.ndarray],
    hours: List[float],
    tau_hours: float,
) -> np.ndarray:
    """Exponential weights w ~ exp(-dt/tau), nan-safe mean."""
    if not layers:
        raise ValueError("empty layers")
    weights = np.array([math.exp(-h / max(tau_hours, 1e-6)) for h in hours], dtype=np.float32)
    weights /= weights.sum()

    acc = np.zeros_like(layers[0], dtype=np.float64)
    wsum = np.zeros_like(layers[0], dtype=np.float64)
    for w, arr in zip(weights, layers):
        m = np.isfinite(arr)
        acc[m] += w * arr[m].astype(np.float64)
        wsum[m] += w
    out = np.full_like(layers[0], np.nan, dtype=np.float64)
    ok = wsum > 1e-9
    out[ok] = acc[ok] / wsum[ok]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Displacement (pixels on reference grid)
# ---------------------------------------------------------------------------


def flow_farneback(prev_u8: np.ndarray, cur_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    p = prev_u8.astype(np.float32)
    c = cur_u8.astype(np.float32)
    p = cv2.GaussianBlur(p, (5, 5), 0)
    c = cv2.GaussianBlur(c, (5, 5), 0)
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    u = flow[..., 0].astype(np.float32)
    v = flow[..., 1].astype(np.float32)
    return u, v


def flow_phase_shift_fallback(prev_u8: np.ndarray, cur_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from skimage.registration import phase_cross_correlation

        scale = 4
        p = prev_u8[::scale, ::scale].astype(np.float32)
        c = cur_u8[::scale, ::scale].astype(np.float32)
        shift, _, _ = phase_cross_correlation(p, c, upsample_factor=2, normalization="phase")
        du = float(shift[0]) * scale
        dv = float(shift[1]) * scale
    except Exception:
        du, dv = 0.0, 0.0
    h, w = cur_u8.shape
    u = np.full((h, w), du, dtype=np.float32)
    v = np.full((h, w), dv, dtype=np.float32)
    return u, v


def estimate_pixel_flow(prev_blend: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (u, v) in pixels: displacement from prev_blend toward current."""
    try:
        return flow_farneback(prev_blend, current)
    except Exception:
        return flow_phase_shift_fallback(prev_blend, current)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class SceneRecord:
    path: Path
    stem: str
    t_start: datetime
    t_end: Optional[datetime]
    bounds: Tuple[float, float, float, float]
    crs: Any


def index_scenes(paths: List[Path]) -> List[SceneRecord]:
    out: List[SceneRecord] = []
    for p in paths:
        stem = p.stem
        try:
            t0, t1 = parse_s1_acquisition_times(stem)
        except ValueError:
            continue
        with rasterio.open(p) as ds:
            rec = SceneRecord(
                path=p,
                stem=stem,
                t_start=t0,
                t_end=t1,
                bounds=tuple(ds.bounds),
                crs=ds.crs,
            )
        out.append(rec)
    out.sort(key=lambda r: r.t_start)
    return out


def pick_predecessors(
    ref: SceneRecord,
    all_scenes: List[SceneRecord],
    num_prev: int,
    min_overlap: float,
    max_hours: Optional[float],
) -> List[SceneRecord]:
    candidates: List[SceneRecord] = []
    for s in all_scenes:
        if s.path == ref.path:
            continue
        if s.t_start >= ref.t_start:
            continue
        iou = geo_bounds_iou(ref.bounds, ref.crs, s.bounds, s.crs)
        if iou < min_overlap:
            continue
        dt_h = hours_between(ref.t_start, s.t_start)
        if max_hours is not None and dt_h > max_hours:
            continue
        candidates.append(s)
    candidates.sort(key=lambda x: x.t_start, reverse=True)
    return candidates[:num_prev]


def build_drift_for_reference(
    ref: SceneRecord,
    predecessors: List[SceneRecord],
    num_classes: int,
    tau_hours: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    with rasterio.open(ref.path) as ref_ds:
        cur = ref_ds.read(1)
        h, w = cur.shape
        transform = ref_ds.transform
        crs = ref_ds.crs
        ref_nodata = ref_ds.nodata

    if not predecessors:
        meta = {"predecessors": [], "note": "no valid previous scenes; drift_prev zeros, flow zeros"}
        zeros = np.zeros((h, w), dtype=np.uint8)
        zf = np.zeros((h, w), dtype=np.float32)
        return zeros, zf, zf, meta

    layers: List[np.ndarray] = []
    dts: List[float] = []
    pred_meta: List[Dict[str, Any]] = []

    for s in predecessors:
        warped = reproject_band_to_reference(
            s.path,
            transform,
            crs,
            h,
            w,
            nodata=ref_nodata,
        )
        valid = np.isfinite(warped)
        warped[~valid] = np.nan
        layers.append(warped)
        dts.append(hours_between(ref.t_start, s.t_start))
        pred_meta.append(
            {
                "path": str(s.path),
                "t_start": s.t_start.isoformat(),
                "hours_before_ref": dts[-1],
            }
        )

    blended = temporal_blend(layers, dts, tau_hours=tau_hours)
    codes = class_codes(num_classes)
    drift_prev = snap_to_codebook(np.nan_to_num(blended, nan=0.0), codes)

    u, v = estimate_pixel_flow(drift_prev.astype(np.uint8), cur.astype(np.uint8))

    meta = {
        "reference": str(ref.path),
        "ref_t_start": ref.t_start.isoformat(),
        "predecessors": pred_meta,
        "tau_hours": tau_hours,
        "num_classes": num_classes,
    }
    return drift_prev, u, v, meta


def write_outputs(
    ref_path: Path,
    drift_prev: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    meta: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ref_path.stem

    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()

    # drift_prev: single band uint8
    prev_profile = profile.copy()
    prev_profile.update(count=1, dtype="uint8", nodata=0)
    out_prev = output_dir / f"{stem}_drift_prev.tif"
    with rasterio.open(out_prev, "w", **prev_profile) as dst:
        dst.write(drift_prev.astype(np.uint8), 1)

    # drift_uv: 2 bands float32
    uv_profile = profile.copy()
    uv_profile.update(count=2, dtype="float32", nodata=None)
    out_uv = output_dir / f"{stem}_drift_uv.tif"
    with rasterio.open(out_uv, "w", **uv_profile) as dst:
        dst.write(u.astype(np.float32), 1)
        dst.write(v.astype(np.float32), 2)

    out_json = output_dir / f"{stem}_drift_meta.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out_prev.name}, {out_uv.name}, {out_json.name}")


def list_scene_tifs(root: Path, pattern: str) -> List[Path]:
    paths = sorted(root.rglob(pattern))
    return [p for p in paths if p.is_file()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", type=Path, required=True, help="Root folder containing GeoTIFF scenes (searched recursively)")
    p.add_argument("--glob", type=str, default="*.tif", help="Glob relative to scenes-dir (default *.tif)")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write *_drift_prev.tif, *_drift_uv.tif, *_drift_meta.json")
    p.add_argument("--num-prev", type=int, default=4, help="Max number of previous acquisitions to blend")
    p.add_argument("--min-overlap", type=float, default=0.02, help="Min geographic IoU (WGS84 approx) with reference")
    p.add_argument("--max-hours", type=float, default=None, help="Ignore predecessors older than this many hours")
    p.add_argument("--tau-hours", type=float, default=48.0, help="Temporal decay for exponential blend (hours)")
    p.add_argument("--num-classes", type=int, default=9, help="Ice class count for snapping blended values to codebook")
    p.add_argument("--only-stem", type=str, default="", help="If set, process only files whose stem contains this substring")
    p.add_argument("--limit", type=int, default=0, help="Process at most N reference scenes (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tifs = list_scene_tifs(args.scenes_dir, args.glob)
    if args.only_stem:
        tifs = [p for p in tifs if args.only_stem in p.stem]
    scenes = index_scenes(tifs)
    if not scenes:
        raise SystemExit("No scenes with parseable S1 datetimes in filenames.")

    print(f"Indexed {len(scenes)} scenes with valid timestamps.")

    n_done = 0
    for ref in scenes:
        preds = pick_predecessors(ref, scenes, args.num_prev, args.min_overlap, args.max_hours)
        drift_prev, u, v, meta = build_drift_for_reference(ref, preds, args.num_classes, args.tau_hours)
        write_outputs(ref.path, drift_prev, u, v, meta, args.output_dir)
        n_done += 1
        if args.limit and n_done >= args.limit:
            break

    print(f"Done. Processed {n_done} reference scene(s).")


if __name__ == "__main__":
    main()
