from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from prepare import (
    convert_source_to_single_band,
    ensure_dir,
    list_tifs,
    make_drift_field,
    rasterize_land,
    read_tif,
    write_tif,
    warp_nearest,
)


DEFAULT_MASK_SUFFIXES: tuple[str, ...] = (
    "_class0_mask",
    "_binary_mask",
    "_mask",
    "-mask",
    " mask",
)


@dataclass(frozen=True)
class PreparedGeoTIFFRecord:
    image_path: str
    mask_path: str
    output_path: str
    band_names: List[str]
    hole_pixels: int


@dataclass(frozen=True)
class ExternalMaskBatchResult:
    output_dir: str
    records: List[PreparedGeoTIFFRecord]
    unmatched_images: List[str]
    unused_masks: List[str]


class ExternalMaskGeoTIFFPreparer:
    """
    Готовит model-ready GeoTIFF из пар `image + external mask`.

    По умолчанию итоговый порядок полос для RGB IceClass:
    1..N. исходные полосы изображения
    N+1. `masked_image` (single-band представление, зануленное по hole-mask)
    N+2. `hole_mask` (0/255, где 255 означает пиксели для заполнения моделью)
    N+3. `land_mask` (0/1)
    N+4. `drift_prev` (single-band proxy предыдущего кадра)

    Для трехканального входа и `save_drift_prev=True` это дает 7 полос.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        output_dir: str,
        *,
        land_vector: Optional[str] = None,
        land_raster: Optional[str] = None,
        num_classes: int = 9,
        save_drift_prev: bool = True,
        mask_suffixes: Sequence[str] = DEFAULT_MASK_SUFFIXES,
    ) -> None:
        self.image_dir = Path(image_dir).resolve()
        self.mask_dir = Path(mask_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.land_vector = land_vector
        self.land_raster = land_raster
        self.num_classes = num_classes
        self.save_drift_prev = save_drift_prev
        self.mask_suffixes = tuple(sorted({suffix.lower() for suffix in mask_suffixes}, key=len, reverse=True))

    def prepare_all(self, *, overwrite: bool = False) -> ExternalMaskBatchResult:
        ensure_dir(self.output_dir)
        image_files = list_tifs(self.image_dir)
        mask_files = list_tifs(self.mask_dir)
        if not image_files:
            raise RuntimeError(f"No .tif files found in image_dir={self.image_dir}")
        if not mask_files:
            raise RuntimeError(f"No .tif files found in mask_dir={self.mask_dir}")

        mask_lookup = self._build_mask_lookup(mask_files)
        records: List[PreparedGeoTIFFRecord] = []
        unmatched_images: List[str] = []
        used_mask_paths: set[str] = set()

        for image_path in image_files:
            mask_path = self._match_mask_for_image(image_path, mask_lookup)
            if mask_path is None:
                unmatched_images.append(str(image_path))
                continue

            used_mask_paths.add(str(mask_path))
            output_path = self.output_dir / f"{image_path.stem}_model_input.tif"
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_path}. Pass overwrite=True to replace it."
                )
            records.append(self.prepare_pair(image_path, mask_path, output_path))

        unused_masks = [str(path) for path in mask_files if str(path) not in used_mask_paths]
        return ExternalMaskBatchResult(
            output_dir=str(self.output_dir),
            records=records,
            unmatched_images=unmatched_images,
            unused_masks=unused_masks,
        )

    def prepare_pair(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        output_path: str | Path,
    ) -> PreparedGeoTIFFRecord:
        image_path = Path(image_path).resolve()
        mask_path = Path(mask_path).resolve()
        output_path = Path(output_path).resolve()

        with rasterio.open(image_path) as image_src, rasterio.open(mask_path) as mask_src:
            self._validate_shared_grid(image_src, mask_src, image_path, mask_path)

        image_arr, image_meta = read_tif(image_path)
        mask_arr, _ = read_tif(mask_path)
        if image_arr.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 source image for prepare-like pipeline, got {image_arr.dtype} in {image_path}"
            )
        if mask_arr.ndim != 3 or mask_arr.shape[0] < 1:
            raise ValueError(f"Expected mask GeoTIFF with at least one band, got shape {mask_arr.shape}")

        hole_mask = self._mask_to_hole_mask(mask_arr[0])
        original_single_band = convert_source_to_single_band(image_arr, num_classes=self.num_classes)
        masked_image = original_single_band.copy()
        masked_image[hole_mask > 0] = 0
        land_mask = self._build_land_mask(image_meta, original_single_band.shape)

        stacked_bands = [band.astype(np.uint8) for band in image_arr]
        band_names = [f"source_band_{index}" for index in range(1, image_arr.shape[0] + 1)]
        stacked_bands.append(masked_image)
        band_names.append("masked_image")
        stacked_bands.append(hole_mask)
        band_names.append("hole_mask")
        stacked_bands.append(land_mask)
        band_names.append("land_mask")

        if self.save_drift_prev:
            seed = abs(hash((image_path.stem, mask_path.stem))) % (2**32)
            max_shift = max(6, min(original_single_band.shape) // 24)
            u, v = make_drift_field(
                original_single_band.shape[0],
                original_single_band.shape[1],
                max_shift=max_shift,
                seed=seed,
            )
            drift_prev = warp_nearest(original_single_band, u, v)
            stacked_bands.append(drift_prev.astype(np.uint8))
            band_names.append("drift_prev")

        stacked = np.stack(stacked_bands, axis=0).astype(np.uint8)
        out_meta = image_meta.copy()
        out_meta.update(count=stacked.shape[0], dtype="uint8", nodata=0)
        write_tif(output_path, stacked, out_meta, dtype="uint8", nodata=0)

        with rasterio.open(output_path, "r+") as dst:
            for band_index, band_name in enumerate(band_names, start=1):
                dst.set_band_description(band_index, band_name)
            dst.update_tags(
                source_image=str(image_path),
                source_mask=str(mask_path),
                external_mask_true_value="255",
                hole_mask_semantics="255 means pixels to fill by the model",
                band_order=",".join(band_names),
            )

        return PreparedGeoTIFFRecord(
            image_path=str(image_path),
            mask_path=str(mask_path),
            output_path=str(output_path),
            band_names=band_names,
            hole_pixels=int(np.count_nonzero(hole_mask == 255)),
        )

    def _build_mask_lookup(self, mask_files: Sequence[Path]) -> dict[str, List[Path]]:
        lookup: dict[str, List[Path]] = {}
        for mask_path in mask_files:
            normalized = self._normalize_stem(mask_path.stem, is_mask=True)
            lookup.setdefault(normalized, []).append(mask_path)
        return lookup

    def _match_mask_for_image(self, image_path: Path, mask_lookup: dict[str, List[Path]]) -> Optional[Path]:
        normalized_image_stem = self._normalize_stem(image_path.stem, is_mask=False)
        candidates = mask_lookup.get(normalized_image_stem, [])
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple mask candidates for {image_path.name}: {[str(candidate) for candidate in candidates]}"
            )
        return candidates[0]

    def _normalize_stem(self, stem: str, *, is_mask: bool) -> str:
        normalized = stem.lower().strip()
        if is_mask:
            for suffix in self.mask_suffixes:
                if normalized.endswith(suffix):
                    normalized = normalized[: -len(suffix)]
                    break
        return normalized.rstrip("_- ")

    def _validate_shared_grid(
        self,
        image_src: rasterio.io.DatasetReader,
        mask_src: rasterio.io.DatasetReader,
        image_path: Path,
        mask_path: Path,
    ) -> None:
        same_grid = (
            image_src.width == mask_src.width
            and image_src.height == mask_src.height
            and str(image_src.crs) == str(mask_src.crs)
            and np.allclose(tuple(image_src.transform)[:6], tuple(mask_src.transform)[:6], atol=1e-9)
        )
        if not same_grid:
            raise ValueError(
                "Image and mask must share width/height/CRS/transform, "
                f"got image={image_path} mask={mask_path}"
            )

    def _mask_to_hole_mask(self, mask_band: np.ndarray) -> np.ndarray:
        return np.where(mask_band == 255, 255, 0).astype(np.uint8)

    def _build_land_mask(self, image_meta: dict, shape_hw: tuple[int, int]) -> np.ndarray:
        if self.land_vector:
            return rasterize_land(self.land_vector, image_meta, shape_hw).astype(np.uint8)
        if not self.land_raster:
            return np.zeros(shape_hw, dtype=np.uint8)

        land_raster_path = Path(self.land_raster).resolve()
        height, width = shape_hw
        with rasterio.open(land_raster_path) as src:
            same_grid = (
                src.width == width
                and src.height == height
                and str(src.crs) == str(image_meta["crs"])
                and np.allclose(tuple(src.transform)[:6], tuple(image_meta["transform"])[:6], atol=1e-9)
            )
            if same_grid:
                land_mask = src.read(1)
            else:
                land_mask = np.zeros(shape_hw, dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=land_mask,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=image_meta["transform"],
                    dst_crs=image_meta["crs"],
                    resampling=Resampling.nearest,
                )
        return (land_mask > 0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--land-vector", default=None)
    parser.add_argument("--land-raster", default=None)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--no-drift-prev", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preparer = ExternalMaskGeoTIFFPreparer(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        land_vector=args.land_vector,
        land_raster=args.land_raster,
        num_classes=args.num_classes,
        save_drift_prev=not args.no_drift_prev,
    )
    result = preparer.prepare_all(overwrite=args.overwrite)
    print(f"Prepared pairs: {len(result.records)}")
    print(f"Unmatched images: {len(result.unmatched_images)}")
    print(f"Unused masks: {len(result.unused_masks)}")
    for record in result.records:
        print(f"- {record.output_path} | bands={record.band_names} | hole_pixels={record.hole_pixels}")


if __name__ == "__main__":
    main()
