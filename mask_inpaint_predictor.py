"""Predict ice classes inside holes and save GeoTIFF.

This module provides a class wrapper around best.pt (trained by train_unet2.py).
It supports:
1) Building a 7-channel input from dataset folders (masked/masks/land_mask/drift_prev).
2) Running model inference.
3) Writing a single-band uint8 GeoTIFF with the same profile as a reference TIFF.

Output is an encoded class map:
- outside hole: original masked values are preserved
- inside hole: model prediction is inserted
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from train_unet2 import ResNet34UNet


class IceMaskInpaintPredictor:
    """Inference wrapper for mask-based inpainting on 7-channel input."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        num_classes: Optional[int] = None,
        use_amp: bool = True,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        self.num_classes = int(num_classes) if num_classes is not None else self._infer_num_classes(ckpt)
        self.output_classes = self.num_classes + 1
        self.class_codes = np.rint(np.linspace(1, 255, self.num_classes)).astype(np.uint8)

        self.model = ResNet34UNet(in_channels=7, num_classes=self.output_classes).to(self.device)
        state = ckpt.get("model", ckpt)
        cleaned = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.eval()

    @staticmethod
    def _infer_num_classes(ckpt: Dict) -> int:
        if "args" in ckpt and isinstance(ckpt["args"], dict) and "num_classes" in ckpt["args"]:
            return int(ckpt["args"]["num_classes"])
        state = ckpt.get("model", ckpt)
        head_key = next((k for k in state if k.endswith("head.2.weight")), None)
        if head_key is None:
            raise RuntimeError("Cannot infer num_classes from checkpoint.")
        return int(state[head_key].shape[0]) - 1

    @staticmethod
    def _read_tif(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    @staticmethod
    def _read_time_norm(dataset_root: Path, split: str, version: str, sample_id: str) -> float:
        samples_csv = dataset_root / "samples.csv"
        with open(samples_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split and row["version"] == version and row["sample_id"] == sample_id:
                    return float(row["time_norm"])
        raise ValueError(f"sample_id={sample_id} not found in {samples_csv}")

    def build_7ch_from_dataset(
        self,
        dataset_root: str | Path,
        split: str,
        version: str,
        sample_id: str,
        time_norm: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Path]:
        """Build input tensor [7,H,W] from dataset files and return hole_mask + reference path."""
        root = Path(dataset_root)
        base = root / split / version

        masked_path = base / "masked" / f"{sample_id}.tif"
        hole_path = base / "masks" / f"{sample_id}.tif"
        land_path = base / "land_mask" / f"{sample_id}.tif"
        drift_path = base / "drift_prev" / f"{sample_id}.tif"

        masked = self._read_tif(masked_path)
        hole = (self._read_tif(hole_path) > 0).astype(np.float32)
        land = (self._read_tif(land_path) > 0).astype(np.float32)
        drift = self._read_tif(drift_path).astype(np.float32) if drift_path.exists() else np.zeros_like(masked, dtype=np.float32)

        if time_norm is None:
            time_norm = self._read_time_norm(root, split, version, sample_id)

        masked_f = masked.astype(np.float32) / 255.0
        drift_f = drift / 255.0

        h, w = masked.shape
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        x_ch = np.repeat(xx, h, axis=0)
        y_ch = np.repeat(yy, w, axis=1)
        time_ch = np.full((h, w), np.float32(time_norm), dtype=np.float32)

        x7 = np.stack([masked_f, hole, land, drift_f, x_ch, y_ch, time_ch], axis=0)
        return x7, hole, hole_path

    def predict_encoded_map(self, x7: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        """Return single-band uint8 encoded class map."""
        if x7.ndim != 3 or x7.shape[0] != 7:
            raise ValueError(f"x7 must be [7,H,W], got {x7.shape}")

        x = torch.from_numpy(x7.astype(np.float32)).unsqueeze(0).to(self.device)
        amp_ok = self.use_amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.no_grad():
            if amp_ok:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = self.model(x)
            else:
                logits = self.model(x)
            pred_class = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)

        pred_encoded = np.zeros_like(pred_class, dtype=np.uint8)
        for cid in range(1, self.num_classes + 1):
            pred_encoded[pred_class == cid] = self.class_codes[cid - 1]

        masked_base = x7[0]
        if masked_base.max() <= 1.0:
            masked_encoded = np.clip(masked_base * 255.0, 0, 255).astype(np.uint8)
        else:
            masked_encoded = np.clip(masked_base, 0, 255).astype(np.uint8)

        hole = hole_mask > 0.5
        restored = masked_encoded.copy()
        restored[hole] = pred_encoded[hole]
        return restored

    @staticmethod
    def save_single_band_tif(image_u8: np.ndarray, output_tif: str | Path, reference_tif: str | Path) -> None:
        output_tif = Path(output_tif)
        output_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(reference_tif) as src:
            profile = src.profile.copy()

        profile.update(count=1, dtype="uint8")
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(image_u8.astype(np.uint8), 1)

    def predict_from_dataset_sample_to_tif(
        self,
        dataset_root: str | Path,
        split: str,
        version: str,
        sample_id: str,
        output_tif: str | Path,
        time_norm: Optional[float] = None,
    ) -> Path:
        """Full pipeline: read sample -> predict in holes -> save GeoTIFF."""
        x7, hole, reference_tif = self.build_7ch_from_dataset(
            dataset_root=dataset_root,
            split=split,
            version=version,
            sample_id=sample_id,
            time_norm=time_norm,
        )
        restored = self.predict_encoded_map(x7, hole)
        self.save_single_band_tif(restored, output_tif=output_tif, reference_tif=reference_tif)
        return Path(output_tif)

    @staticmethod
    def visualize_prediction(
        original_u8: np.ndarray,
        hole_mask: np.ndarray,
        prediction_u8: np.ndarray,
        output_png: str | Path,
        title: str = "Mask inpainting result",
    ) -> Path:
        """Save a 3-panel visualization: original image, mask, prediction."""
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_u8, cmap="gray", interpolation="nearest")
        axes[0].set_title("Original image")

        axes[1].imshow(hole_mask > 0.5, cmap="gray", interpolation="nearest")
        axes[1].set_title("Mask")

        axes[2].imshow(prediction_u8, cmap="gray", interpolation="nearest")
        axes[2].set_title("Model prediction")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return output_png


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--dataset-root", default="dataset/dataset")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--version", required=True, help="e.g. vertical_tri / horizontal_tri / irregular_shape")
    p.add_argument("--sample-id", required=True)
    p.add_argument("--output-tif", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output-vis", default="", help="Optional PNG path to save visualization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictor = IceMaskInpaintPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_classes=args.num_classes,
        use_amp=not args.no_amp,
    )
    out = predictor.predict_from_dataset_sample_to_tif(
        dataset_root=args.dataset_root,
        split=args.split,
        version=args.version,
        sample_id=args.sample_id,
        output_tif=args.output_tif,
    )
    print(f"Saved: {out}")

    if args.output_vis:
        x7, hole, _ = predictor.build_7ch_from_dataset(
            dataset_root=args.dataset_root,
            split=args.split,
            version=args.version,
            sample_id=args.sample_id,
        )
        original_u8 = np.clip(x7[0] * 255.0, 0, 255).astype(np.uint8)
        prediction_u8 = predictor.predict_encoded_map(x7, hole)
        vis_path = predictor.visualize_prediction(
            original_u8=original_u8,
            hole_mask=hole,
            prediction_u8=prediction_u8,
            output_png=args.output_vis,
            title=f"{args.split}/{args.version}/{args.sample_id}",
        )
        print(f"Saved visualization: {vis_path}")


if __name__ == "__main__":
    main()

