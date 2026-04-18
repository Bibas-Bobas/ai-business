"""Train an ice-class gap-filling model on the prepared VIZARD dataset.

Tuned for a 4x A100 box. Launch with torchrun:

    torchrun --standalone --nproc_per_node=4 train.py \
        --dataset-root dataset --epochs 40 --batch-size 32 \
        --class-values 1,33,64,96,128,160,192,223

Single-GPU fallback:

    python train.py --dataset-root dataset --epochs 40 --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Distributed / logging
# ---------------------------------------------------------------------------


@dataclass
class DistConfig:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    enabled: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistConfig:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        enabled = world_size > 1
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if enabled and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                init_method="env://",
            )
    else:
        rank, local_rank, world_size, enabled = 0, 0, 1, False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistConfig(rank=rank, local_rank=local_rank, world_size=world_size, device=device, enabled=enabled)


def setup_logger(log_dir: Path, rank: int) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        f"%(asctime)s | r{rank} | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(log_dir / "train.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def barrier(cfg: DistConfig) -> None:
    if cfg.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor, cfg: DistConfig) -> torch.Tensor:
    if cfg.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def resolve_dataset_root(path: str) -> Path:
    base = Path(path)
    for candidate in (base, base / "dataset", base / "data"):
        if (candidate / "samples.csv").exists():
            return candidate
    nested = sorted(base.glob("*/samples.csv"))
    if nested:
        return nested[0].parent
    raise FileNotFoundError(f"samples.csv not found under {base}")


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


@dataclass
class SampleRecord:
    sample_id: str
    split: str
    version: str
    time_norm: float
    original_path: Path
    masked_path: Path
    hole_path: Path
    land_path: Path
    drift_path: Optional[Path]


def load_records(dataset_root: Path, split: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    with open(dataset_root / "samples.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            base = dataset_root / row["split"] / row["version"]
            sample_id = row["sample_id"]
            drift = base / "drift_prev" / f"{sample_id}.tif"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    split=row["split"],
                    version=row["version"],
                    time_norm=float(row["time_norm"]),
                    original_path=base / "original" / f"{sample_id}.tif",
                    masked_path=base / "masked" / f"{sample_id}.tif",
                    hole_path=base / "masks" / f"{sample_id}.tif",
                    land_path=base / "land_mask" / f"{sample_id}.tif",
                    drift_path=drift if drift.exists() else None,
                )
            )
    return records


def discover_class_values(dataset_root: Path, cache_path: Path, io_threads: int, show_progress: bool) -> List[int]:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return [int(v) for v in json.load(f)["class_values"]]

    tifs = sorted(dataset_root.glob("*/**/original/*.tif"))
    values: set[int] = set()
    with ThreadPoolExecutor(max_workers=io_threads) as pool:
        for arr in tqdm(pool.map(read_band, tifs, chunksize=8), total=len(tifs), desc="class scan", disable=not show_progress):
            values.update(int(v) for v in np.unique(arr).tolist())
    class_values = sorted(v for v in values if v > 0)
    if not class_values:
        raise RuntimeError("No positive class values found in dataset")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"class_values": class_values}, f, indent=2)
    return class_values


def compute_class_weights(
    records: Sequence[SampleRecord],
    class_values: Sequence[int],
    sample_limit: int,
    io_threads: int,
    show_progress: bool,
) -> torch.Tensor:
    selected = list(records)
    if sample_limit > 0 and len(selected) > sample_limit:
        stride = max(1, len(selected) // sample_limit)
        selected = selected[::stride][:sample_limit]

    num = len(class_values)
    lut = np.zeros(256, dtype=np.int64)
    for i, v in enumerate(class_values):
        lut[int(v)] = i

    def _count(rec: SampleRecord) -> np.ndarray:
        arr = read_band(rec.original_path)
        land = read_band(rec.land_path)
        water = land == 0
        mapped = lut[arr[water]] if np.any(water) else lut[arr]
        return np.bincount(mapped.reshape(-1), minlength=num).astype(np.float64)

    counts = np.zeros(num, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=io_threads) as pool:
        for row in tqdm(pool.map(_count, selected), total=len(selected), desc="class weights", disable=not show_progress):
            counts += row

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    weights = np.power(freq, -0.5)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.35, 4.0)
    present = counts > counts.sum() * 1e-5
    weights = np.where(present, weights, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


class VizardDataset(Dataset):
    """Sample = 6 spatial channels + optional drift channel → class id per pixel."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        class_values: Sequence[int],
        augment: bool,
        use_drift: bool,
    ) -> None:
        if not records:
            raise RuntimeError("Empty record list")
        self.records = list(records)
        self.augment = augment
        self.use_drift = use_drift
        self.class_lut = np.zeros(256, dtype=np.int64)
        for i, v in enumerate(class_values):
            self.class_lut[int(v)] = i
        self._coord_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _coords(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (h, w)
        if key not in self._coord_cache:
            y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
            x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing="ij")
            self._coord_cache[key] = (xx, yy)
        return self._coord_cache[key]

    @staticmethod
    def _apply_aug(arrays: List[np.ndarray]) -> List[np.ndarray]:
        if np.random.rand() < 0.5:
            arrays = [np.ascontiguousarray(a[..., ::-1]) for a in arrays]
        if np.random.rand() < 0.5:
            arrays = [np.ascontiguousarray(a[..., ::-1, :]) for a in arrays]
        k = np.random.randint(0, 4)
        if k:
            arrays = [np.ascontiguousarray(np.rot90(a, k, axes=(-2, -1))) for a in arrays]
        return arrays

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        try:
            original = read_band(rec.original_path)
            masked = read_band(rec.masked_path)
            hole = (read_band(rec.hole_path) > 0).astype(np.float32)
            land = (read_band(rec.land_path) > 0).astype(np.float32)
            if self.use_drift and rec.drift_path is not None:
                drift = read_band(rec.drift_path).astype(np.float32) / 255.0
            else:
                drift = None

            h, w = original.shape
            xx, yy = self._coords(h, w)
            t_chan = np.full((h, w), rec.time_norm, dtype=np.float32)
            target = self.class_lut[original].astype(np.int64)
            valid = (land < 0.5).astype(np.float32)

            masked_f = masked.astype(np.float32) / 255.0

            if self.augment:
                stack = [masked_f, hole, t_chan, land, target.astype(np.float32), valid]
                if drift is not None:
                    stack.append(drift)
                stack = self._apply_aug(stack)
                masked_f, hole, t_chan, land = stack[0], stack[1], stack[2], stack[3]
                target = stack[4].astype(np.int64)
                valid = stack[5]
                if drift is not None:
                    drift = stack[6]

            channels = [masked_f, hole, xx, yy, t_chan, land]
            if self.use_drift:
                channels.append(drift if drift is not None else np.zeros((h, w), dtype=np.float32))
            image = np.stack(channels, axis=0)

            return {
                "image": torch.from_numpy(image),
                "target": torch.from_numpy(target),
                "hole_mask": torch.from_numpy(hole[None, ...]),
                "valid_mask": torch.from_numpy(valid[None, ...]),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load '{rec.sample_id}' ({rec.original_path})") from e


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def group_norm(ch: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class SE(nn.Module):
    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SE(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.drop(self.act(self.norm2(h))))
        return self.se(h) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.block = ResBlock(out_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.conv(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResBlock(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x.float(), size=skip.shape[-2:], mode="nearest").to(skip.dtype)
        return self.block(torch.cat([x, skip], dim=1))


class IceUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 6, base * 8]
        self.stem = ResBlock(in_channels, c[0])
        self.d1 = Down(c[0], c[1], dropout * 0.5)
        self.d2 = Down(c[1], c[2], dropout * 0.75)
        self.d3 = Down(c[2], c[3], dropout)
        self.d4 = Down(c[3], c[4], dropout)
        self.u4 = Up(c[4], c[3], c[3], dropout)
        self.u3 = Up(c[3], c[2], c[2], dropout * 0.75)
        self.u2 = Up(c[2], c[1], c[1], dropout * 0.5)
        self.u1 = Up(c[1], c[0], c[0], dropout * 0.25)
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        b = self.d4(s3)
        u = self.u4(b, s3)
        u = self.u3(u, s2)
        u = self.u2(u, s1)
        u = self.u1(u, s0)
        return self.head(u)


# ---------------------------------------------------------------------------
# Loss / metrics / scheduler / EMA
# ---------------------------------------------------------------------------


def soft_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    probs = logits.float().softmax(dim=1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    m = mask.float()
    probs, oh = probs * m, oh * m
    inter = (probs * oh).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def seg_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    hole: torch.Tensor,
    valid: torch.Tensor,
    class_weights: torch.Tensor,
    num_classes: int,
    hole_weight: float,
    context_weight: float,
    dice_weight: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pixel_w = valid * (context_weight + hole_weight * hole)
    ce = F.cross_entropy(
        logits, target, weight=class_weights, reduction="none", label_smoothing=label_smoothing
    )
    ce = (ce * pixel_w.squeeze(1)).sum() / pixel_w.sum().clamp_min(1.0)

    dice_full = soft_dice(logits, target, valid, num_classes)
    dice_hole = soft_dice(logits, target, valid * hole, num_classes)
    loss = ce + dice_weight * (0.3 * dice_full + 0.7 * dice_hole)
    parts = {
        "ce": float(ce.detach().item()),
        "dice_full": float(dice_full.detach().item()),
        "dice_hole": float(dice_hole.detach().item()),
    }
    return loss, parts


class ConfMetrics:
    def __init__(self, num_classes: int, device: torch.device) -> None:
        self.num = num_classes
        self.device = device
        self.full = torch.zeros(num_classes * num_classes, dtype=torch.float64, device=device)
        self.hole = torch.zeros(num_classes * num_classes, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, hole: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        v = valid.squeeze(1) > 0.5
        h = v & (hole.squeeze(1) > 0.5)
        self._add(self.full, pred[v], target[v])
        self._add(self.hole, pred[h], target[h])

    def _add(self, dest: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.numel() == 0:
            return
        idx = target.to(torch.int64) * self.num + pred.to(torch.int64)
        dest += torch.bincount(idx, minlength=self.num * self.num).to(torch.float64)

    def sync(self, cfg: DistConfig) -> None:
        all_reduce_sum(self.full, cfg)
        all_reduce_sum(self.hole, cfg)

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, conf in (("full", self.full), ("hole", self.hole)):
            m = conf.view(self.num, self.num).cpu()
            tp = m.diag()
            support = m.sum(dim=1)
            predicted = m.sum(dim=0)
            union = support + predicted - tp
            iou = tp / union.clamp_min(1.0)
            present = support > 0
            acc = (tp.sum() / m.sum().clamp_min(1.0)).item()
            miou = iou[present].mean().item() if present.any() else 0.0
            out[f"{name}_acc"] = float(acc)
            out[f"{name}_miou"] = float(miou)
        return out


class WarmupCosine:
    def __init__(self, opt: torch.optim.Optimizer, warmup: int, total: int, min_scale: float = 0.05) -> None:
        self.opt = opt
        self.warmup = max(1, warmup)
        self.total = max(1, total)
        self.min_scale = min_scale
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / self.warmup
        else:
            p = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
            scale = self.min_scale + (1.0 - self.min_scale) * 0.5 * (1.0 + math.cos(math.pi * p))
        for lr, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = lr * scale


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.lerp_(msd[k].detach(), 1.0 - self.decay)
            else:
                v.copy_(msd[k])


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def autocast_ctx(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[WarmupCosine],
    ema: Optional[EMA],
    cfg: DistConfig,
    class_weights: torch.Tensor,
    num_classes: int,
    amp_dtype: Optional[torch.dtype],
    channels_last: bool,
    hole_weight: float,
    context_weight: float,
    dice_weight: float,
    label_smoothing: float,
    grad_clip: float,
    max_batches: Optional[int],
    phase: str,
    epoch: int,
    logger: logging.Logger,
    show_progress: bool,
) -> Dict[str, float]:
    training = phase == "train"
    model.train(training)
    device = cfg.device
    metrics = ConfMetrics(num_classes, device)
    running = torch.zeros(4, dtype=torch.float64, device=device)  # loss, ce, dice_hole, count

    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    bar = tqdm(
        total=total,
        desc=f"{phase} {epoch + 1}",
        leave=False,
        disable=not (show_progress and cfg.is_main),
        dynamic_ncols=True,
    )

    start = time.time()
    data_t = time.time()
    step_idx = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            data_time = time.time() - data_t
            batch = move_batch(batch, device)
            image = batch["image"]
            if channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            step_start = time.time()
            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(device, amp_dtype):
                logits = model(image)
                loss, parts = seg_loss(
                    logits=logits,
                    target=batch["target"],
                    hole=batch["hole_mask"],
                    valid=batch["valid_mask"],
                    class_weights=class_weights,
                    num_classes=num_classes,
                    hole_weight=hole_weight,
                    context_weight=context_weight,
                    dice_weight=dice_weight,
                    label_smoothing=label_smoothing,
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(unwrap(model))

            metrics.update(logits.detach(), batch["target"], batch["valid_mask"], batch["hole_mask"])
            running[0] += float(loss.detach().item())
            running[1] += parts["ce"]
            running[2] += parts["dice_hole"]
            running[3] += 1.0

            step_idx += 1
            step_time = time.time() - step_start

            if cfg.is_main:
                bar.update(1)
                avg_loss = float(running[0].item()) / float(running[3].item())
                avg_hole = float(running[2].item()) / float(running[3].item())
                postfix = {"loss": f"{avg_loss:.3f}", "dice_hole": f"{avg_hole:.3f}",
                           "data": f"{data_time:.2f}s", "step": f"{step_time:.2f}s"}
                if training:
                    postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.1e}"
                bar.set_postfix(postfix)

            if max_batches is not None and step_idx >= max_batches:
                break
            data_t = time.time()

    bar.close()

    all_reduce_sum(running, cfg)
    metrics.sync(cfg)
    count = max(1.0, float(running[3].item()))
    summary = {
        "loss": float(running[0].item()) / count,
        "ce": float(running[1].item()) / count,
        "dice_hole": float(running[2].item()) / count,
    }
    summary.update(metrics.summary())
    if cfg.is_main:
        logger.info(f"{phase} ep={epoch + 1} time={time.time() - start:.1f}s " +
                    " ".join(f"{k}={v:.4f}" for k, v in summary.items()))
    return summary


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="dataset")
    p.add_argument("--output-dir", default="runs/ice_unet")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=0, help="Per-rank workers (0 = auto)")
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--io-threads", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--auto-scale-lr", action="store_true")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--hole-weight", type=float, default=4.0)
    p.add_argument("--context-weight", type=float, default=0.35)
    p.add_argument("--dice-weight", type=float, default=0.75)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--class-weight-scan-limit", type=int, default=1500)
    p.add_argument("--class-values", type=str, default=None,
                   help="Comma-separated class values; skips scanning if given")
    p.add_argument("--disable-drift", action="store_true")
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--max-train-batches", type=int, default=0, help="Cap train batches (smoke)")
    p.add_argument("--max-val-batches", type=int, default=0, help="Cap val batches (smoke)")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def auto_num_workers(requested: int, world_size: int) -> int:
    if requested > 0:
        return requested
    cpu = os.cpu_count() or 8
    return min(24, max(2, cpu // max(1, world_size) - 1))


def main() -> None:
    args = parse_args()
    cfg = init_distributed()

    torch.manual_seed(args.seed + cfg.rank)
    np.random.seed(args.seed + cfg.rank)

    if cfg.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    if cfg.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(cfg)
    logger = setup_logger(output_dir, cfg.rank)

    # AMP setup: prefer bf16 on A100, fp16 elsewhere, off if disabled
    if args.disable_amp or cfg.device.type != "cuda":
        amp_dtype = None
    elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    channels_last = cfg.device.type == "cuda"
    num_workers = auto_num_workers(args.num_workers, cfg.world_size)
    show_progress = not args.no_progress

    if cfg.is_main:
        logger.info(
            f"world={cfg.world_size} device={cfg.device} amp={amp_dtype} "
            f"channels_last={channels_last} workers/rank={num_workers}"
        )
        if cfg.device.type == "cuda":
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  gpu[{i}] {props.name} {props.total_memory / 1024 ** 3:.0f} GiB")

    dataset_root = resolve_dataset_root(args.dataset_root)
    if cfg.is_main:
        logger.info(f"dataset_root={dataset_root}")

    class_cache = output_dir / "class_values.json"
    if args.class_values:
        class_values = [int(v) for v in args.class_values.split(",") if v.strip()]
        if cfg.is_main:
            class_cache.write_text(json.dumps({"class_values": class_values}))
    else:
        if cfg.is_main:
            class_values = discover_class_values(dataset_root, class_cache, args.io_threads, show_progress)
        barrier(cfg)
        if not cfg.is_main:
            class_values = discover_class_values(dataset_root, class_cache, args.io_threads, False)
    num_classes = len(class_values)
    if cfg.is_main:
        logger.info(f"class_values={class_values} num_classes={num_classes}")

    train_records = load_records(dataset_root, "train")
    val_records = load_records(dataset_root, "val")
    if cfg.is_main:
        logger.info(f"records: train={len(train_records)} val={len(val_records)}")

    train_ds = VizardDataset(train_records, class_values, augment=True, use_drift=not args.disable_drift)
    val_ds = VizardDataset(val_records, class_values, augment=False, use_drift=not args.disable_drift)

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if cfg.enabled:
        train_sampler = DistributedSampler(train_ds, cfg.world_size, cfg.rank, shuffle=True, drop_last=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, cfg.world_size, cfg.rank, shuffle=False, drop_last=False, seed=args.seed)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=cfg.device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **loader_kwargs)

    if cfg.is_main:
        logger.info(
            f"batches/rank train={len(train_loader)} val={len(val_loader)} "
            f"world_batch={args.batch_size * cfg.world_size}"
        )

    sample = train_ds[0]
    in_channels = sample["image"].shape[0]
    if cfg.is_main:
        logger.info(f"in_channels={in_channels} base={args.base_channels} dropout={args.dropout}")

    model = IceUNet(in_channels, num_classes, base=args.base_channels, dropout=args.dropout).to(cfg.device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    if cfg.is_main:
        logger.info(f"params={n_params / 1e6:.2f}M")

    ema = EMA(model, args.ema_decay)

    if cfg.enabled:
        model = DDP(
            model,
            device_ids=[cfg.local_rank] if cfg.device.type == "cuda" else None,
            output_device=cfg.local_rank if cfg.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    effective_lr = args.lr * cfg.world_size if args.auto_scale_lr else args.lr
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=effective_lr, weight_decay=args.weight_decay, betas=(0.9, 0.99),
        fused=cfg.device.type == "cuda",
    )
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_epochs * max(1, len(train_loader)))
    scheduler = WarmupCosine(optimizer, warmup_steps, total_steps)

    if cfg.is_main:
        class_weights_cpu = compute_class_weights(
            train_records, class_values, args.class_weight_scan_limit, args.io_threads, show_progress,
        )
        logger.info(f"class_weights={[round(v, 3) for v in class_weights_cpu.tolist()]}")
    else:
        class_weights_cpu = torch.ones(num_classes)
    class_weights = class_weights_cpu.to(cfg.device)
    if cfg.enabled:
        dist.broadcast(class_weights, src=0)

    max_train = args.max_train_batches if args.max_train_batches > 0 else None
    max_val = args.max_val_batches if args.max_val_batches > 0 else None

    history: List[Dict[str, float]] = []
    best_metric = -1.0
    epochs_without_improvement = 0
    t_total = time.time()

    if cfg.is_main:
        config_dump = {k: (v if isinstance(v, (int, float, str, bool, list)) or v is None else str(v))
                       for k, v in vars(args).items()}
        config_dump.update({
            "world_size": cfg.world_size, "effective_lr": effective_lr, "num_classes": num_classes,
            "class_values": class_values, "amp_dtype": str(amp_dtype), "in_channels": in_channels,
            "params_M": round(n_params / 1e6, 3),
        })
        (output_dir / "config.json").write_text(json.dumps(config_dump, indent=2))

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = run_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, ema=ema,
            cfg=cfg, class_weights=class_weights, num_classes=num_classes,
            amp_dtype=amp_dtype, channels_last=channels_last,
            hole_weight=args.hole_weight, context_weight=args.context_weight,
            dice_weight=args.dice_weight, label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip, max_batches=max_train,
            phase="train", epoch=epoch, logger=logger, show_progress=show_progress,
        )
        val_metrics = run_epoch(
            model=ema.shadow, loader=val_loader, optimizer=None, scheduler=None, ema=None,
            cfg=cfg, class_weights=class_weights, num_classes=num_classes,
            amp_dtype=amp_dtype, channels_last=channels_last,
            hole_weight=args.hole_weight, context_weight=args.context_weight,
            dice_weight=args.dice_weight, label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip, max_batches=max_val,
            phase="val", epoch=epoch, logger=logger, show_progress=show_progress,
        )

        if cfg.is_main:
            row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))

            torch.save({"model": unwrap(model).state_dict(),
                        "ema": ema.shadow.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "class_values": class_values}, output_dir / "last.pt")

        score = val_metrics["hole_miou"]
        if score > best_metric:
            best_metric = score
            epochs_without_improvement = 0
            if cfg.is_main:
                torch.save({"model": unwrap(model).state_dict(),
                            "ema": ema.shadow.state_dict(),
                            "epoch": epoch,
                            "best_metric": best_metric,
                            "class_values": class_values}, output_dir / "best.pt")
                logger.info(f"  new best hole_miou={best_metric:.4f}")
        else:
            epochs_without_improvement += 1
            if cfg.is_main:
                logger.info(f"  no improvement ({epochs_without_improvement}/{args.early_stop_patience})")

        barrier(cfg)
        if epochs_without_improvement >= args.early_stop_patience:
            if cfg.is_main:
                logger.info(f"Early stop at epoch {epoch + 1}")
            break

    if cfg.is_main:
        logger.info(f"Done in {(time.time() - t_total) / 60:.1f} min. Best hole_miou={best_metric:.4f}")

    if cfg.enabled and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
