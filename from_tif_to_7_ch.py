"""Build a 7-channel GeoTIFF from source TIFF + hole-mask TIFF.

Channel order matches the training pipeline from `prepare.py`:
0. masked_image
1. hole_mask
2. land_mask
3. drift_prev
4. x_coord
5. y_coord
6. time

The converter works window-by-window, so large scenes can be processed without
materializing the whole 7-channel tensor in RAM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window


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


@dataclass
class PreviewData:
    source_display: np.ndarray
    source_single_band: np.ndarray
    hole_mask: np.ndarray
    channel_data: np.ndarray
    channel_name: str


class SevenChannelTifBuilder:
    """Convert source TIFF + mask TIFF into a 7-channel float32 GeoTIFF."""

    CHANNEL_NAMES = {
        0: "masked_image",
        1: "hole_mask",
        2: "land_mask",
        3: "drift_prev",
        4: "x_coord",
        5: "y_coord",
        6: "time",
    }

    def __init__(self, num_classes: int = 9, tile_size: int = 1024) -> None:
        self.num_classes = num_classes
        self.tile_size = tile_size

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rgb_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a.astype(np.float32) - b.astype(np.float32)
        return np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))

    def _rgb_to_class_map(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[0] < 3:
            raise ValueError(f"Expected RGB array shaped [3,H,W] or more, got {rgb.shape}")

        img = np.transpose(rgb[:3], (1, 2, 0)).astype(np.uint8)
        h, w, _ = img.shape
        class_ids = sorted(ICE_PALETTE.keys())
        colors = np.array([ICE_PALETTE[cid] for cid in class_ids], dtype=np.uint8)
        flat = img.reshape(-1, 3)
        out = np.zeros((flat.shape[0],), dtype=np.uint8)

        exact = np.zeros((flat.shape[0],), dtype=bool)
        for cid, color in zip(class_ids, colors):
            hit = np.all(flat == color, axis=1)
            out[hit] = cid
            exact |= hit

        rest = ~exact
        if np.any(rest):
            dists = np.stack([self._rgb_distance(flat[rest], color) for color in colors], axis=1)
            nearest = dists.argmin(axis=1)
            out[rest] = np.array(class_ids, dtype=np.uint8)[nearest]

        return out.reshape(h, w)

    def _encode_classes_u8(self, class_map: np.ndarray) -> np.ndarray:
        values = np.rint(np.linspace(1, 255, self.num_classes)).astype(np.uint8)
        out = np.zeros_like(class_map, dtype=np.uint8)
        for cid in range(1, self.num_classes + 1):
            out[class_map == cid] = values[min(cid - 1, len(values) - 1)]
        return out

    def _convert_source_to_single_band(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            raise ValueError(f"Expected source TIFF shaped [C,H,W], got {arr.shape}")
        if arr.shape[0] == 1:
            return arr[0].astype(np.uint8)
        if arr.shape[0] >= 3:
            class_map = self._rgb_to_class_map(arr[:3])
            return self._encode_classes_u8(class_map)
        raise ValueError(f"Unsupported source channel count: {arr.shape[0]}")

    @staticmethod
    def _iter_windows(height: int, width: int, tile_size: int) -> Iterator[Window]:
        for row_off in range(0, height, tile_size):
            for col_off in range(0, width, tile_size):
                yield Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(tile_size, width - col_off),
                    height=min(tile_size, height - row_off),
                )

    @staticmethod
    def _normalize_u8(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32) / 255.0

    @staticmethod
    def _coords_for_window(
        window: Window,
        full_height: int,
        full_width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.arange(int(window.row_off), int(window.row_off + window.height), dtype=np.float32)
        cols = np.arange(int(window.col_off), int(window.col_off + window.width), dtype=np.float32)

        if full_width > 1:
            x_values = (cols / (full_width - 1)) * 2.0 - 1.0
        else:
            x_values = np.zeros_like(cols, dtype=np.float32)

        if full_height > 1:
            y_values = (rows / (full_height - 1)) * 2.0 - 1.0
        else:
            y_values = np.zeros_like(rows, dtype=np.float32)

        x_grid = np.repeat(x_values[None, :], len(rows), axis=0).astype(np.float32)
        y_grid = np.repeat(y_values[:, None], len(cols), axis=1).astype(np.float32)
        return x_grid, y_grid

    def _build_window_tensor(
        self,
        source_window: np.ndarray,
        hole_window: np.ndarray,
        window: Window,
        full_height: int,
        full_width: int,
        time_value: float,
        land_window: Optional[np.ndarray] = None,
        drift_window: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        source_u8 = self._convert_source_to_single_band(source_window)
        hole = (hole_window > 0).astype(np.float32)
        land = (
            (land_window > 0).astype(np.float32)
            if land_window is not None
            else np.zeros_like(hole, dtype=np.float32)
        )
        drift = (
            self._normalize_u8(drift_window)
            if drift_window is not None
            else np.zeros_like(hole, dtype=np.float32)
        )

        source_norm = self._normalize_u8(source_u8)
        masked = source_norm.copy()
        masked[hole > 0.5] = 0.0

        x_grid, y_grid = self._coords_for_window(window, full_height=full_height, full_width=full_width)
        time_channel = np.full_like(hole, np.float32(time_value), dtype=np.float32)

        return np.stack([masked, hole, land, drift, x_grid, y_grid, time_channel], axis=0)

    def convert_to_7ch_tif(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        output_path: str | Path,
        *,
        land_mask_path: Optional[str | Path] = None,
        drift_prev_path: Optional[str | Path] = None,
        time_value: float = 1.0,
    ) -> Path:
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        output_path = Path(output_path)
        self._ensure_parent(output_path)

        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
            if src_img.width != src_mask.width or src_img.height != src_mask.height:
                raise ValueError(
                    "Image and mask must have the same spatial size: "
                    f"image=({src_img.height}, {src_img.width}) "
                    f"mask=({src_mask.height}, {src_mask.width})"
                )

            src_land = rasterio.open(land_mask_path) if land_mask_path else None
            src_drift = rasterio.open(drift_prev_path) if drift_prev_path else None
            try:
                profile = src_img.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=7,
                    dtype="float32",
                    compress="lzw",
                    tiled=True,
                    BIGTIFF="IF_SAFER",
                    nodata=None,
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    for window in self._iter_windows(src_img.height, src_img.width, self.tile_size):
                        source_window = src_img.read(window=window)
                        hole_window = src_mask.read(1, window=window)
                        land_window = src_land.read(1, window=window) if src_land is not None else None
                        drift_window = src_drift.read(1, window=window) if src_drift is not None else None

                        tensor_window = self._build_window_tensor(
                            source_window=source_window,
                            hole_window=hole_window,
                            land_window=land_window,
                            drift_window=drift_window,
                            window=window,
                            full_height=src_img.height,
                            full_width=src_img.width,
                            time_value=time_value,
                        )
                        dst.write(tensor_window.astype(np.float32), window=window)
            finally:
                if src_land is not None:
                    src_land.close()
                if src_drift is not None:
                    src_drift.close()

        return output_path

    def build_preview(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        *,
        channel_index: int = 0,
        time_value: float = 1.0,
        max_side: int = 1024,
    ) -> PreviewData:
        image_path = Path(image_path)
        mask_path = Path(mask_path)
        channel_name = self.CHANNEL_NAMES.get(channel_index, f"channel_{channel_index}")

        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
            scale = max(src_img.height, src_img.width) / float(max_side)
            if scale > 1.0:
                preview_height = max(1, int(round(src_img.height / scale)))
                preview_width = max(1, int(round(src_img.width / scale)))
            else:
                preview_height = src_img.height
                preview_width = src_img.width

            source_preview = src_img.read(
                out_shape=(src_img.count, preview_height, preview_width),
                resampling=rasterio.enums.Resampling.nearest,
            )
            mask_preview = src_mask.read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=rasterio.enums.Resampling.nearest,
            )

        source_single = self._convert_source_to_single_band(source_preview)
        preview_window = Window(0, 0, preview_width, preview_height)
        preview_tensor = self._build_window_tensor(
            source_window=source_preview,
            hole_window=mask_preview,
            window=preview_window,
            full_height=preview_height,
            full_width=preview_width,
            time_value=time_value,
        )

        if source_preview.shape[0] >= 3:
            source_display = np.transpose(source_preview[:3], (1, 2, 0)).astype(np.uint8)
        else:
            source_display = source_single

        return PreviewData(
            source_display=source_display,
            source_single_band=source_single,
            hole_mask=(mask_preview > 0).astype(np.uint8),
            channel_data=preview_tensor[channel_index],
            channel_name=channel_name,
        )

    @staticmethod
    def save_preview(preview: PreviewData, output_png: str | Path) -> Path:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        if preview.source_display.ndim == 3:
            axes[0].imshow(preview.source_display, interpolation="nearest")
        else:
            axes[0].imshow(preview.source_display, cmap="gray", interpolation="nearest")
        axes[0].set_title("Input TIFF")

        axes[1].imshow(preview.source_single_band, cmap="gray", interpolation="nearest")
        axes[1].set_title("Input as 1 band")

        axes[2].imshow(preview.hole_mask, cmap="gray", interpolation="nearest")
        axes[2].set_title("Hole mask")

        axes[3].imshow(preview.channel_data, cmap="gray", interpolation="nearest")
        axes[3].set_title(f"7ch: {preview.channel_name}")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(output_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="Source TIFF, either 1-band or RGB-coded 3-band")
    parser.add_argument("--mask-path", required=True, help="Mask TIFF, non-zero values are treated as holes")
    parser.add_argument("--output-path", required=True, help="Output 7-channel float32 GeoTIFF")
    parser.add_argument("--output-vis", default="", help="Optional PNG preview path")
    parser.add_argument("--land-mask-path", default=None, help="Optional land mask TIFF")
    parser.add_argument("--drift-prev-path", default=None, help="Optional previous-frame TIFF")
    parser.add_argument("--time-value", type=float, default=1.0, help="Constant value for time channel")
    parser.add_argument("--num-classes", type=int, default=9, help="Number of encoded classes for RGB source TIFF")
    parser.add_argument("--tile-size", type=int, default=1024, help="Processing tile size for large images")
    parser.add_argument("--preview-channel", type=int, default=0, choices=list(range(7)), help="Which 7-channel band to show in preview")
    parser.add_argument("--preview-max-side", type=int, default=1024, help="Maximum side length for visualization preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = SevenChannelTifBuilder(num_classes=args.num_classes, tile_size=args.tile_size)

    output_path = builder.convert_to_7ch_tif(
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_path=args.output_path,
        land_mask_path=args.land_mask_path,
        drift_prev_path=args.drift_prev_path,
        time_value=args.time_value,
    )
    print(f"Saved 7-channel TIFF: {output_path}")

    if args.output_vis:
        preview = builder.build_preview(
            image_path=args.image_path,
            mask_path=args.mask_path,
            channel_index=args.preview_channel,
            time_value=args.time_value,
            max_side=args.preview_max_side,
        )
        preview_path = builder.save_preview(preview, args.output_vis)
        print(f"Saved preview PNG: {preview_path}")


if __name__ == "__main__":
    main()


