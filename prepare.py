"""Prepare a compact VIZARD training dataset with synthetic drift.

This script creates a dataset of GeoTIFF patches suitable for training an
inpainting model with channels:
- masked_image
- hole_mask
- land_mask
- (x, y, time are generated in the training Dataset on the fly)

It also creates a drift surrogate to imitate ice motion between satellite passes:
- drift_prev: previous-frame proxy produced by warping the original patch with a
  smooth random displacement field.

Output structure
----------------
output_root/
├── train/
│   ├── original/
│   ├── masked/
│   ├── masks/
│   ├── land_mask/
│   └── drift_prev/
├── val/
│   └── ...
├── test/
│   └── ...
└── meta.csv

Key design choices
------------------
- We work with patches, not full scenes, to keep the output bounded.
- The output budget is controlled with --target-gb.
- By default the script generates a moderate number of patches per scene so the
  resulting dataset stays around a few GB, not hundreds of GB.
- RGB-coded ice maps are converted to a single-channel uint8 class map.
- The script supports 16 workers on DataSphere.

Example
-------
python prepare_vizard_training_dataset_v2.py \
  --input-root /path/to/source_tifs \
  --output-root /path/to/dataset \
  --land-vector /path/to/russia.shp \
  --patch-size 384 \
  --target-gb 6 \
  --workers 16 \
  --save-drift-prev
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from PIL import Image, ImageDraw
from tqdm import tqdm


ICE_PALETTE: Dict[int, Tuple[int, int, int]] = {
    1: (0, 100, 255),
    2: (0, 34, 223),
    3: (250, 0, 255),
    4: (0, 250, 0),
    5: (0, 200, 200),
    6: (150, 150, 150),
    7: (171, 243, 255),
    8: (1, 1, 1),
    9: (255, 255, 255),
}


# -------------------------
# IO / conversion
# -------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_tifs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])


def read_tif(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read()  # [C,H,W]
        meta = src.meta.copy()
        meta["transform"] = src.transform
        meta["crs"] = src.crs
    return arr, meta


def write_tif(path: Path, arr: np.ndarray, meta: dict, dtype: str = "uint8", nodata: int = 0) -> None:
    ensure_dir(path.parent)
    if arr.ndim == 2:
        arr = arr[None, ...]
    out_meta = meta.copy()
    out_meta.update(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=dtype,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
        nodata=nodata,
    )
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr.astype(dtype))


def rgb_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))


def rgb_to_class_map(rgb: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] < 3:
        raise ValueError(f"Expected [3,H,W] or more, got {rgb.shape}")

    img = np.transpose(rgb[:3], (1, 2, 0)).astype(np.uint8)
    h, w, _ = img.shape
    class_ids = sorted(palette.keys())
    colors = np.array([palette[c] for c in class_ids], dtype=np.uint8)
    flat = img.reshape(-1, 3)
    out = np.zeros((flat.shape[0],), dtype=np.uint8)

    exact = np.zeros((flat.shape[0],), dtype=bool)
    for cid, color in zip(class_ids, colors):
        hit = np.all(flat == color, axis=1)
        out[hit] = cid
        exact |= hit

    rest = ~exact
    if np.any(rest):
        dists = np.stack([rgb_distance(flat[rest], c) for c in colors], axis=1)
        nearest = dists.argmin(axis=1)
        out[rest] = np.array(class_ids, dtype=np.uint8)[nearest]

    return out.reshape(h, w)


def encode_classes_u8(class_map: np.ndarray, num_classes: int) -> np.ndarray:
    values = np.rint(np.linspace(1, 255, num_classes)).astype(np.uint8)
    out = np.zeros_like(class_map, dtype=np.uint8)
    for cid in range(1, num_classes + 1):
        out[class_map == cid] = values[min(cid - 1, len(values) - 1)]
    return out


def convert_source_to_single_band(arr: np.ndarray, num_classes: int) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {arr.shape}")
    if arr.shape[0] == 1:
        return arr[0].astype(np.uint8)
    if arr.shape[0] >= 3:
        class_map = rgb_to_class_map(arr[:3], ICE_PALETTE)
        return encode_classes_u8(class_map, num_classes=num_classes)
    raise ValueError(f"Unsupported channel count: {arr.shape[0]}")


def crop_pad_2d(arr: np.ndarray, top: int, left: int, size: int, pad_value: int = 0) -> np.ndarray:
    h, w = arr.shape
    out = np.full((size, size), pad_value, dtype=arr.dtype)
    bottom = min(h, top + size)
    right = min(w, left + size)
    out[: bottom - top, : right - left] = arr[top:bottom, left:right]
    return out


# -------------------------
# Land mask
# -------------------------


def rasterize_land(vector_path: Optional[str], meta: dict, shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if vector_path is None:
        return np.zeros((h, w), dtype=np.uint8)

    try:
        import fiona
        from shapely.geometry import shape as shp_shape
    except Exception as e:
        raise RuntimeError("For --land-vector install fiona and shapely, or use --land-raster.") from e

    geoms = []
    with fiona.open(vector_path, "r") as src:
        for feat in src:
            geom = feat.get("geometry")
            if geom is not None:
                geoms.append(shp_shape(geom))

    if not geoms:
        return np.zeros((h, w), dtype=np.uint8)

    return rasterize(
        [(g, 1) for g in geoms],
        out_shape=(h, w),
        transform=meta["transform"],
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


# -------------------------
# Drift synthesis
# -------------------------


def smooth_noise_field(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small_h = max(2, h // 32)
    small_w = max(2, w // 32)
    noise = rng.random((small_h, small_w)).astype(np.float32)
    img = Image.fromarray(np.uint8(noise * 255), mode="L")
    img = img.resize((w, h), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def make_drift_field(h: int, w: int, max_shift: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base_angle = rng.uniform(-np.pi, np.pi)
    magnitude = rng.uniform(2.0, float(max_shift))
    n1 = smooth_noise_field(h, w, seed=seed)
    n2 = smooth_noise_field(h, w, seed=seed + 13)
    yy, xx = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32), np.linspace(-1, 1, w, dtype=np.float32), indexing="ij")
    u = magnitude * np.cos(base_angle) * (0.4 + 0.6 * n1) + 2.0 * yy
    v = magnitude * np.sin(base_angle) * (0.4 + 0.6 * n2) + 2.0 * xx
    return u.astype(np.float32), v.astype(np.float32)


def warp_nearest(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    src_x = np.clip(np.rint(xx - u).astype(np.int32), 0, w - 1)
    src_y = np.clip(np.rint(yy - v).astype(np.int32), 0, h - 1)
    return image[src_y, src_x]


# -------------------------
# Synthetic holes
# -------------------------


def dilate_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask > 0, iterations=iterations).astype(np.uint8)
    except Exception:
        m = (mask > 0).astype(np.uint8)
        for _ in range(iterations):
            pad = np.pad(m, 1, mode="constant")
            neigh = [pad[i : i + m.shape[0], j : j + m.shape[1]] for i in range(3) for j in range(3)]
            m = np.maximum.reduce(neigh)
        return m.astype(np.uint8)


def make_hole_mask(h: int, w: int, version: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    if version == "horizontal_tri":
        apex_y = rng.integers(h // 4, 3 * h // 4)
        apex_x = rng.integers(0, w)
        slope = rng.uniform(0.35, 0.85)
        if rng.random() < 0.5:
            mask = (yy >= apex_y + slope * np.abs(xx - apex_x))
        else:
            mask = (yy <= apex_y - slope * np.abs(xx - apex_x))

    elif version == "vertical_tri":
        apex_y = rng.integers(0, h)
        apex_x = rng.integers(w // 4, 3 * w // 4)
        slope = rng.uniform(0.35, 0.85)
        if rng.random() < 0.5:
            mask = (xx >= apex_x + slope * np.abs(yy - apex_y))
        else:
            mask = (xx <= apex_x - slope * np.abs(yy - apex_y))

    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(rng.integers(2, 5)):
            cy = rng.integers(0, h)
            cx = rng.integers(0, w)
            ry = rng.integers(max(16, h // 12), max(20, h // 4))
            rx = rng.integers(max(16, w // 12), max(20, w // 4))
            blob = (((yy - cy) ** 2) / (ry * ry + 1e-8) + ((xx - cx) ** 2) / (rx * rx + 1e-8)) <= 1.0
            mask |= blob.astype(np.uint8)
        for _ in range(rng.integers(1, 3)):
            thick = rng.integers(max(6, min(h, w) // 40), max(12, min(h, w) // 10))
            angle = rng.uniform(-1.1, 1.1)
            cy = rng.integers(0, h)
            cx = rng.integers(0, w)
            dist = np.abs((xx - cx) * math.sin(angle) - (yy - cy) * math.cos(angle))
            mask |= (dist <= thick).astype(np.uint8)

    return dilate_mask(mask.astype(np.uint8), iterations=int(rng.integers(1, 4)))


# -------------------------
# Budgeting
# -------------------------


def estimate_patches_per_scene(target_gb: float, patch_size: int, n_scenes: int, save_drift_prev: bool) -> int:
    """Conservative patch count per scene.

    We deliberately overshoot the raw bytes a bit because compressed GeoTIFFs
    are much smaller than the uncompressed estimate.
    """
    target_bytes = target_gb * (1024 ** 3)
    bytes_per_sample = patch_size * patch_size * (4 + (1 if save_drift_prev else 0))
    if bytes_per_sample <= 0 or n_scenes <= 0:
        return 1
    samples_total = target_bytes / bytes_per_sample
    per_scene = int(max(4, min(16, samples_total / (n_scenes * 3))))  # /3 because we create 3 versions
    return max(4, per_scene)


# -------------------------
# Worker task
# -------------------------


@dataclass
class Task:
    src_path: str
    out_root: str
    split: str
    version: str
    patch_size: int
    patches_per_scene: int
    num_classes: int
    land_vector: Optional[str]
    land_raster: Optional[str]
    save_drift_prev: bool


def process_task(task: Task) -> Tuple[bool, str, int, List[dict]]:
    try:
        src_path = Path(task.src_path)
        arr, meta = read_tif(src_path)
        h, w = arr.shape[1], arr.shape[2]
        stem = src_path.stem

        original_full = convert_source_to_single_band(arr, num_classes=task.num_classes)
        if task.land_vector:
            land_full = rasterize_land(task.land_vector, meta, (h, w))
        elif task.land_raster:
            with rasterio.open(task.land_raster) as src:
                if src.width != w or src.height != h:
                    land_full = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
                else:
                    land_full = src.read(1)
            land_full = (land_full > 0).astype(np.uint8)
        else:
            land_full = np.zeros((h, w), dtype=np.uint8)

        # Choose patch anchors. More patches if the source is large.
        stride = max(64, task.patch_size // 2)
        anchors = [(y, x) for y in range(0, max(1, h - task.patch_size + 1), stride)
                         for x in range(0, max(1, w - task.patch_size + 1), stride)]
        if not anchors:
            anchors = [(0, 0)]

        rng = np.random.default_rng(abs(hash((stem, task.version))) % (2**32))
        if len(anchors) > task.patches_per_scene:
            chosen = rng.choice(len(anchors), size=task.patches_per_scene, replace=False)
            anchors = [anchors[i] for i in chosen]
        else:
            # If there are too few anchors, sample random repeats to reach the cap.
            while len(anchors) < task.patches_per_scene:
                anchors.append(anchors[rng.integers(0, len(anchors))])

        out_base = Path(task.out_root) / task.split / task.version
        for sub in ["original", "masked", "masks", "land_mask"]:
            ensure_dir(out_base / sub)
        if task.save_drift_prev:
            ensure_dir(out_base / "drift_prev")

        bytes_est = 0
        rows = []
        for i, (top, left) in enumerate(anchors):
            seed = abs(hash((stem, task.version, i))) % (2**32)
            patch_orig = crop_pad_2d(original_full, top, left, task.patch_size, pad_value=0)
            patch_land = crop_pad_2d(land_full, top, left, task.patch_size, pad_value=0)

            # imitate the previous overpass (drifted ice field)
            max_shift = max(6, task.patch_size // 24)
            u, v = make_drift_field(task.patch_size, task.patch_size, max_shift=max_shift, seed=seed)
            drift_prev = warp_nearest(patch_orig, u, v) if task.save_drift_prev else None

            hole = make_hole_mask(task.patch_size, task.patch_size, task.version, seed=seed + 17)
            patch_masked = patch_orig.copy()
            patch_masked[hole > 0] = 0

            sample_id = f"{stem}_p{i:03d}"
            out_meta = meta.copy()
            out_meta.update(
                driver="GTiff",
                height=task.patch_size,
                width=task.patch_size,
                count=1,
                dtype="uint8",
                compress="lzw",
                tiled=True,
                BIGTIFF="IF_SAFER",
                nodata=0,
            )
            write_tif(out_base / "original" / f"{sample_id}.tif", patch_orig, out_meta)
            write_tif(out_base / "masked" / f"{sample_id}.tif", patch_masked, out_meta)
            write_tif(out_base / "masks" / f"{sample_id}.tif", hole, out_meta)
            write_tif(out_base / "land_mask" / f"{sample_id}.tif", patch_land, out_meta)
            if drift_prev is not None:
                write_tif(out_base / "drift_prev" / f"{sample_id}.tif", drift_prev, out_meta)

            # metadata row for later time-aware training
            rows.append(
                {
                    "split": task.split,
                    "version": task.version,
                    "sample_id": sample_id,
                    "source_file": src_path.name,
                    "patch_index": i,
                    "time_norm": round(i / max(1, task.patches_per_scene - 1), 6),
                    "top": top,
                    "left": left,
                    "patch_size": task.patch_size,
                    "drift_prev": int(task.save_drift_prev),
                    "drift_uv": 0,
                }
            )

            bytes_est += task.patch_size * task.patch_size * (4 + (1 if task.save_drift_prev else 0))

        return True, str(src_path), bytes_est, rows
    except Exception as e:
        return False, f"{task.src_path}: {e}", 0, []


# -------------------------
# CLI / main
# -------------------------


def split_files(files: List[Path], seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test = [files[i] for i in idx[:n_test]]
    val = [files[i] for i in idx[n_test : n_test + n_val]]
    train = [files[i] for i in idx[n_test + n_val :]]
    return train, val, test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--land-vector", default=None)
    p.add_argument("--land-raster", default=None)
    p.add_argument("--patch-size", type=int, default=384)
    p.add_argument("--target-gb", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument("--versions", nargs="+", default=["horizontal_tri", "irregular_shape", "vertical_tri"])
    p.add_argument("--save-drift-prev", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    files = list_tifs(input_root)
    if not files:
        raise RuntimeError(f"No .tif files found in {input_root}")

    train_files, val_files, test_files = split_files(files, seed=args.seed)
    patches_per_scene = estimate_patches_per_scene(
        target_gb=args.target_gb,
        patch_size=args.patch_size,
        n_scenes=len(files),
        save_drift_prev=args.save_drift_prev,
    )

    print(f"Found {len(files)} tif files")
    print(f"Split: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"Patch size: {args.patch_size}")
    print(f"Target disk budget: ~{args.target_gb:.1f} GB")
    print(f"Patches per scene: {patches_per_scene}")
    print(f"Versions: {args.versions}")
    print(f"Workers: {args.workers}")

    tasks: List[Task] = []
    for split_name, split_files_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for src in split_files_list:
            for version in args.versions:
                tasks.append(
                    Task(
                        src_path=str(src),
                        out_root=str(output_root),
                        split=split_name,
                        version=version,
                        patch_size=args.patch_size,
                        patches_per_scene=patches_per_scene,
                        num_classes=args.num_classes,
                        land_vector=args.land_vector,
                        land_raster=args.land_raster,
                        save_drift_prev=args.save_drift_prev,
                    )
                )

    meta_path = output_root / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["input_root", str(input_root)])
        writer.writerow(["patch_size", args.patch_size])
        writer.writerow(["target_gb", args.target_gb])
        writer.writerow(["patches_per_scene", patches_per_scene])
        writer.writerow(["versions", ",".join(args.versions)])
        writer.writerow(["save_drift_prev", int(args.save_drift_prev)])
        writer.writerow(["train_files", len(train_files)])
        writer.writerow(["val_files", len(val_files)])
        writer.writerow(["test_files", len(test_files)])

    all_rows: List[dict] = []
    ok = 0
    failed = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_task, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            success, msg, bytes_est, rows = fut.result()
            if success:
                ok += 1
                total_bytes += bytes_est
                all_rows.extend(rows)
            else:
                failed += 1
                print(f"[FAIL] {msg}")

    # append sample rows
    samples_csv = output_root / "samples.csv"
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "version", "sample_id", "source_file", "patch_index", "time_norm", "top", "left", "patch_size", "drift_prev", "drift_uv"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Done. ok={ok}, failed={failed}")
    print(f"Meta: {meta_path}")
    print(f"Samples: {samples_csv}")
    print(f"Approx raw bytes generated (before compression): {total_bytes}")


if __name__ == "__main__":
    main()
