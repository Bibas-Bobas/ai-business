"""Evaluate and visualize predictions of the trained ResNet34-UNet on the VIZARD dataset.

Works with checkpoints produced by train_unet2.py (key "model" contains the weights,
key "args" stores the training CLI, which includes num_classes).

CLI usage
---------
    # Compute metrics on the whole val split
    python inference.py --checkpoint best.pt --dataset-root dataset --split val --metrics

    # Visualize 16 random val samples into runs/inference/vis/
    python inference.py --checkpoint best.pt --dataset-root dataset --split val \
        --num-samples 16 --output-dir runs/inference

    # Specific samples by id
    python inference.py --checkpoint best.pt --dataset-root dataset --split val \
        --sample-ids S1A_..._p001 S1A_..._p017

Jupyter usage
-------------
    from inference import load_model, evaluate_split, visualize_samples
    model, meta = load_model("best.pt")
    evaluate_split(model, meta, "dataset", "val")
    visualize_samples(model, meta, "dataset", "val", n=8, output_dir="runs/vis")
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_unet2 import (
    ResNet34UNet,
    VizardDataset,
    decode_encoded_classes_u8,
    read_tif_1ch,
)


# ---------------------------------------------------------------------------
# Ice class palette (IMO), matches prepare.py::ICE_PALETTE
# ---------------------------------------------------------------------------


CLASS_NAMES: Dict[int, str] = {
    0: "no class / background",
    1: "open water",
    2: "nilas",
    3: "young ice",
    4: "thin first-year",
    5: "medium/thick first-year",
    6: "level first-year",
    7: "melting first-year",
    8: "class 8",
    9: "class 9",
}

CLASS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
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


def class_ids_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """Convert class-id map [H,W] into RGB image [H,W,3] with explicit palette."""
    out = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cid, rgb in CLASS_RGB.items():
        out[class_map == cid] = rgb
    # Unexpected class ids are highlighted in red to catch mapping bugs quickly.
    known = np.isin(class_map, np.array(list(CLASS_RGB.keys()), dtype=np.int64))
    out[~known] = (255, 0, 0)
    return out


def make_legend_handles(num_classes_inclusive: int) -> List[Patch]:
    handles = []
    for cid in range(num_classes_inclusive):
        r, g, b = CLASS_RGB.get(cid, (127, 127, 127))
        handles.append(
            Patch(facecolor=(r / 255.0, g / 255.0, b / 255.0), edgecolor="black",
                  label=f"{cid}: {CLASS_NAMES.get(cid, 'class ' + str(cid))}")
        )
    return handles


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class ModelMeta:
    num_classes: int
    output_classes: int
    in_channels: int
    device: torch.device


def _extract_num_classes(ckpt: Dict) -> int:
    if "args" in ckpt and isinstance(ckpt["args"], dict) and "num_classes" in ckpt["args"]:
        return int(ckpt["args"]["num_classes"])
    state = ckpt.get("model", ckpt)
    head_key = next((k for k in state if k.endswith("head.2.weight")), None)
    if head_key is None:
        head_key = next((k for k in state if k.endswith("head.2.bias")), None)
    if head_key is None:
        raise RuntimeError("Cannot infer num_classes from checkpoint: missing head weights")
    return int(state[head_key].shape[0]) - 1


def load_model(checkpoint_path: str | Path, device: Optional[str | torch.device] = None) -> Tuple[torch.nn.Module, ModelMeta]:
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = _extract_num_classes(ckpt)
    in_channels = 7
    model = ResNet34UNet(in_channels=in_channels, num_classes=num_classes + 1).to(device)

    state = ckpt.get("model", ckpt)
    cleaned = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    meta = ModelMeta(
        num_classes=num_classes,
        output_classes=num_classes + 1,
        in_channels=in_channels,
        device=device,
    )
    print(f"[load_model] checkpoint={checkpoint_path} num_classes={num_classes} device={device}")
    if "best_metric" in ckpt:
        print(f"[load_model] best_metric (training): {ckpt['best_metric']:.4f}")
    if "epoch" in ckpt:
        print(f"[load_model] saved at epoch {ckpt['epoch']}")
    return model, meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def make_input_tensor(sample: Dict[str, np.ndarray]) -> torch.Tensor:
    masked = sample["masked"].astype(np.float32) / 255.0
    hole = sample["hole"].astype(np.float32)
    land = sample["land"].astype(np.float32)
    drift = sample["drift"].astype(np.float32) / 255.0 if "drift" in sample else np.zeros_like(masked)

    h, w = masked.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    x_ch = np.repeat(xx, h, axis=0)
    y_ch = np.repeat(yy, w, axis=1)
    t_ch = np.full((h, w), np.float32(sample.get("time_norm", 0.0)), dtype=np.float32)

    x = np.stack([masked, hole, land, drift, x_ch, y_ch, t_ch], axis=0)
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def predict_tensor(model: torch.nn.Module, x: torch.Tensor, device: torch.device, use_amp: bool = True) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    amp_ok = use_amp and device.type == "cuda"
    dtype = torch.bfloat16 if (amp_ok and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16
    if amp_ok:
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
    else:
        logits = model(x)
    return logits.float().argmax(dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_confusion(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, num_classes_inclusive: int) -> torch.Tensor:
    pred = pred[valid].to(torch.int64)
    target = target[valid].to(torch.int64)
    idx = target * num_classes_inclusive + pred
    bins = torch.bincount(idx, minlength=num_classes_inclusive ** 2)
    return bins.view(num_classes_inclusive, num_classes_inclusive)


def confusion_metrics(conf: torch.Tensor) -> Dict[str, float | List[float]]:
    conf = conf.double()
    tp = conf.diag()
    support = conf.sum(dim=1)
    predicted = conf.sum(dim=0)
    union = support + predicted - tp
    iou = tp / union.clamp_min(1.0)
    acc = (tp.sum() / conf.sum().clamp_min(1.0)).item()
    present = support > 0
    miou = iou[present].mean().item() if present.any() else 0.0
    per_class_iou = [float(v) if bool(p) else float("nan") for v, p in zip(iou.tolist(), present.tolist())]
    per_class_support = support.tolist()
    return {"acc": float(acc), "miou": float(miou), "per_class_iou": per_class_iou, "per_class_support": per_class_support}


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    meta: ModelMeta,
    dataset_root: str | Path,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 4,
    use_amp: bool = True,
) -> Dict[str, float | List[float]]:
    dataset = VizardDataset(Path(dataset_root), split=split, num_classes=meta.num_classes, use_drift_prev=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=meta.device.type == "cuda",
    )
    num_out = meta.output_classes
    conf_full = torch.zeros(num_out, num_out, dtype=torch.int64, device=meta.device)
    conf_hole = torch.zeros(num_out, num_out, dtype=torch.int64, device=meta.device)

    for x, target, ignore, hole in tqdm(loader, desc=f"eval {split}", dynamic_ncols=True):
        x = x.to(meta.device, non_blocking=True)
        target = target.to(meta.device, non_blocking=True)
        ignore = ignore.to(meta.device, non_blocking=True)
        hole = hole.to(meta.device, non_blocking=True)
        pred = predict_tensor(model, x, meta.device, use_amp=use_amp)

        valid = ignore == 0
        in_hole = valid & (hole > 0.5)
        conf_full += compute_confusion(pred, target, valid, num_out).to(meta.device)
        conf_hole += compute_confusion(pred, target, in_hole, num_out).to(meta.device)

    full_metrics = confusion_metrics(conf_full.cpu())
    hole_metrics = confusion_metrics(conf_hole.cpu())

    print(f"\n=== Metrics on split={split} ({len(dataset)} samples) ===")
    print(f"full: acc={full_metrics['acc']:.4f} miou={full_metrics['miou']:.4f}")
    print(f"hole: acc={hole_metrics['acc']:.4f} miou={hole_metrics['miou']:.4f}")
    print("per-class IoU (full | hole) support:")
    for cid in range(num_out):
        name = CLASS_NAMES.get(cid, f"class {cid}")
        iou_f = full_metrics["per_class_iou"][cid]
        iou_h = hole_metrics["per_class_iou"][cid]
        sup = full_metrics["per_class_support"][cid]
        iou_f_s = "  nan" if iou_f != iou_f else f"{iou_f:.4f}"
        iou_h_s = "  nan" if iou_h != iou_h else f"{iou_h:.4f}"
        print(f"  [{cid}] {name:28s}  full={iou_f_s}  hole={iou_h_s}  support={int(sup):>10d}")
    return {"full": full_metrics, "hole": hole_metrics}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _read_record(dataset_root: Path, row: Dict[str, str]) -> Dict[str, np.ndarray]:
    base = dataset_root / row["split"] / row["version"]
    sid = row["sample_id"]
    out = {
        "sample_id": sid,
        "split": row["split"],
        "version": row["version"],
        "time_norm": float(row["time_norm"]),
        "original": read_tif_1ch(base / "original" / f"{sid}.tif"),
        "masked": read_tif_1ch(base / "masked" / f"{sid}.tif"),
        "hole": (read_tif_1ch(base / "masks" / f"{sid}.tif") > 0).astype(np.uint8),
        "land": (read_tif_1ch(base / "land_mask" / f"{sid}.tif") > 0).astype(np.uint8),
    }
    drift_path = base / "drift_prev" / f"{sid}.tif"
    if drift_path.exists():
        out["drift"] = read_tif_1ch(drift_path)
    return out


def _load_samples_csv(dataset_root: Path, split: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(dataset_root / "samples.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)
    return rows


def visualize_samples(
    model: torch.nn.Module,
    meta: ModelMeta,
    dataset_root: str | Path,
    split: str = "val",
    n: int = 8,
    output_dir: str | Path = "runs/inference/vis",
    sample_ids: Optional[Sequence[str]] = None,
    seed: int = 42,
    use_amp: bool = True,
    show: bool = False,
) -> List[Path]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_samples_csv(dataset_root, split)
    if sample_ids:
        wanted = set(sample_ids)
        selected = [r for r in rows if r["sample_id"] in wanted]
        missing = wanted - {r["sample_id"] for r in selected}
        if missing:
            print(f"[visualize] not found: {sorted(missing)}")
    else:
        rng = random.Random(seed)
        selected = rng.sample(rows, min(n, len(rows)))

    legend_handles = make_legend_handles(meta.output_classes)

    saved_paths: List[Path] = []
    for row in tqdm(selected, desc="visualize", dynamic_ncols=True):
        sample = _read_record(dataset_root, row)
        target = decode_encoded_classes_u8(sample["original"], meta.num_classes)
        x = make_input_tensor(sample)
        pred = predict_tensor(model, x, meta.device, use_amp=use_amp).squeeze(0).cpu().numpy().astype(np.int64)

        hole = sample["hole"].astype(bool)
        land = sample["land"].astype(bool)
        composite = target.copy()
        composite[hole] = pred[hole]

        valid = ~land
        correct = (pred == target) & valid
        error = np.zeros_like(target, dtype=np.int64)
        error[valid & ~correct] = 1
        error[valid & ~correct & hole] = 2

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        axes[0].imshow(class_ids_to_rgb(target), interpolation="nearest")
        axes[0].set_title("Ground truth (target)")

        masked_show = target.copy()
        masked_show[hole] = 0
        axes[1].imshow(class_ids_to_rgb(masked_show), interpolation="nearest")
        axes[1].set_title("Masked input (with holes)")

        hole_vis = np.zeros_like(target, dtype=np.uint8)
        hole_vis[hole] = 1
        hole_vis[land] = 2
        axes[2].imshow(hole_vis, cmap=ListedColormap(["#1e1e1e", "#ff3355", "#8b6b00"]), vmin=0, vmax=2, interpolation="nearest")
        axes[2].set_title("Masks (red=hole, brown=land)")

        axes[3].imshow(class_ids_to_rgb(pred), interpolation="nearest")
        axes[3].set_title("Prediction (full)")

        axes[4].imshow(class_ids_to_rgb(composite), interpolation="nearest")
        axes[4].set_title("Restored (GT outside hole, pred inside)")

        axes[5].imshow(error, cmap=ListedColormap(["#1e1e1e", "#ffa500", "#ff3355"]), vmin=0, vmax=2, interpolation="nearest")
        axes[5].set_title("Errors (orange=valid err, red=hole err)")

        total_pixels = int(valid.sum())
        total_correct = int(correct.sum())
        pixel_acc = total_correct / max(1, total_pixels)
        hole_valid = valid & hole
        hole_correct = int((correct & hole).sum())
        hole_total = int(hole_valid.sum())
        hole_acc = hole_correct / max(1, hole_total)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            f"{sample['sample_id']}  [{row['split']}/{row['version']}]\n"
            f"pixel_acc={pixel_acc:.3f}  hole_acc={hole_acc:.3f}  hole_pixels={hole_total}",
            fontsize=11,
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(5, meta.output_classes),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])

        out_path = output_dir / f"{sample['sample_id']}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        saved_paths.append(out_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"[visualize] saved {len(saved_paths)} images to {output_dir}")
    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--dataset-root", default="dataset")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default="runs/inference")
    p.add_argument("--num-samples", type=int, default=8, help="Random samples to visualize (ignored if --sample-ids given)")
    p.add_argument("--sample-ids", nargs="*", default=None, help="Explicit sample ids to visualize")
    p.add_argument("--metrics", action="store_true", help="Compute metrics on the whole split")
    p.add_argument("--no-vis", action="store_true", help="Skip visualization")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, meta = load_model(args.checkpoint, device=args.device)
    use_amp = not args.no_amp

    if args.metrics:
        evaluate_split(
            model, meta, args.dataset_root, split=args.split,
            batch_size=args.batch_size, num_workers=args.num_workers, use_amp=use_amp,
        )

    if not args.no_vis:
        visualize_samples(
            model, meta, args.dataset_root, split=args.split,
            n=args.num_samples, sample_ids=args.sample_ids,
            output_dir=Path(args.output_dir) / "vis",
            seed=args.seed, use_amp=use_amp, show=False,
        )


if __name__ == "__main__":
    main()
