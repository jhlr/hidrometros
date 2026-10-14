"""Identical training protocol for the four models (assignment Part 4)."""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from typing import cast
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from hidrometros import data
from hidrometros.models import (
    EARLY_STOP_PATIENCE,
    LABEL_SMOOTHING,
    PROTOCOL,
    SEED,
    ModelSpec,
    backbone_parameters,
    build_model,
    head_parameters,
    set_backbone_frozen,
    train_mode,
)
from hidrometros.reproducibility import set_seed, worker_init_fn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HISTORY_COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "seconds"]
CKPT_DIR: Path = data.ROOT / "checkpoints"
NUM_WORKERS = 4

UInt8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Variant:
    """Where one experiment (base pipeline, extra challenge, smoke test) reads and writes."""

    tag: str
    index_csv: Path
    history_dir: Path
    ckpt_root: Path
    reports_dir: Path
    results_csv: Path

    def history(self, name: str) -> Path:
        return self.history_dir / f"history_{name}.csv"


BASE = Variant(
    "base",
    data.DIGITS_DIR / "index.csv",
    data.ROOT,
    CKPT_DIR,
    data.ROOT / "reports",
    data.ROOT / "results.csv",
)
EXTRA = Variant(
    "extra",
    data.ROOT / "digits_extra" / "index.csv",
    data.ROOT / "extra",
    CKPT_DIR / "extra",
    data.ROOT / "extra" / "reports",
    data.ROOT / "results_extra.csv",
)
SMOKE = Variant(
    "smoke",
    data.DIGITS_DIR / "index.csv",
    data.ROOT / "local" / "smoke",
    data.ROOT / "local" / "smoke" / "ckpt",
    data.ROOT / "local" / "smoke" / "reports",
    data.ROOT / "local" / "smoke" / "results.csv",
)
VARIANTS: dict[str, Variant] = {v.tag: v for v in (BASE, EXTRA, SMOKE)}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_size_for(spec: ModelSpec, device: torch.device) -> int:
    return spec.batch_cuda if device.type == "cuda" else spec.batch_mps_cpu


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def eval_transform(spec: ModelSpec) -> v2.Compose:
    """Resize to the model input and normalize with ImageNet statistics (all four models)."""
    return v2.Compose(
        [
            v2.Resize(spec.input_hw, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(spec: ModelSpec) -> v2.Compose:
    """Protocol augmentation: affine + color jitter, NO flip."""
    return v2.Compose(
        [
            v2.Resize(spec.input_hw, antialias=True),
            v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_crops(index: pd.DataFrame) -> torch.Tensor:
    """Load all crops of an index table into one uint8 tensor (N, 3, 64, 48)."""
    arr = np.stack([np.asarray(Image.open(data.ROOT / p).convert("RGB"), dtype=np.uint8) for p in index["crop_path"]])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class CropDataset(Dataset[tuple[torch.Tensor, int]]):
    """In-memory digit crops with a per-model transform."""

    def __init__(self, images: torch.Tensor, labels: npt.NDArray[np.int64], transform: v2.Compose) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[i]), int(self.labels[i])


def split_index(split: str, index_csv: Path, limit: int | None = None) -> pd.DataFrame:
    """Rows of an index table for one split (optionally a seeded sample of `limit`, for the smoke test)."""
    idx = pd.read_csv(index_csv)
    idx = idx[idx["split"] == split].reset_index(drop=True)
    if limit is not None:
        idx = idx.sample(n=min(limit, len(idx)), random_state=SEED).reset_index(drop=True)
    return idx


def make_loader(
    index: pd.DataFrame, spec: ModelSpec, batch: int, train: bool, images: torch.Tensor | None = None
) -> DataLoader[tuple[torch.Tensor, int]]:
    imgs = images if images is not None else load_crops(index)
    labels = index["digit"].to_numpy(dtype=np.int64)
    ds = CropDataset(imgs, labels, train_transform(spec) if train else eval_transform(spec))
    gen = torch.Generator().manual_seed(SEED)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=train,
        drop_last=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )


@dataclass
class RunState:
    epoch: int = 0  # epochs completed
    best_f1: float = -1.0
    best_epoch: int = 0
    bad_epochs: int = 0


def _rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: dict[str, object]) -> None:
    random.setstate(cast(tuple[object, ...], state["python"]))
    np.random.set_state(cast(tuple[str, npt.NDArray[np.uint32], int, int, float], state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch"]))
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(cast(torch.Tensor, state["mps"]))


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], device: torch.device
) -> tuple[float, float, float, npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Return (loss, acc, f1_macro, y_true, y_pred, confidence) over a loader (plain CE loss)."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, loss_sum = 0, 0.0
    ys: list[torch.Tensor] = []
    ps: list[torch.Tensor] = []
    cs: list[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += float(crit(out, y))
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(dim=1)
        ys.append(y.cpu())
        ps.append(pred.cpu())
        cs.append(conf.cpu())
        total += len(y)
    y_true = torch.cat(ys).numpy().astype(np.int64)
    y_pred = torch.cat(ps).numpy().astype(np.int64)
    conf_all = torch.cat(cs).numpy().astype(np.float32)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(10)), zero_division=0))
    return loss_sum / total, acc, f1, y_true, y_pred, conf_all


def run_training(
    name: str,
    *,
    variant: Variant = BASE,
    resume: bool = False,
    max_epochs: int | None = None,
    limit_train: int | None = None,
    limit_val: int | None = None,
) -> pd.DataFrame:
    """Train one model under the common protocol and return its history table.

    `max_epochs`, `limit_train` and `limit_val` exist only for the smoke test; real runs leave them None.
    """
    spec = PROTOCOL[name]
    device = get_device()
    set_seed(SEED)
    history_path = variant.history(name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = variant.ckpt_root / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = max_epochs or spec.max_epochs
    batch = batch_size_for(spec, device)

    train_idx = split_index("train", variant.index_csv, limit_train)
    val_idx = split_index("val", variant.index_csv, limit_val)
    train_loader = make_loader(train_idx, spec, batch, train=True)
    val_loader = make_loader(val_idx, spec, batch, train=False)

    model = build_model(name, pretrained=spec.pretrained).to(device)
    if spec.lr_backbone is None:
        groups = [{"params": list(model.parameters()), "lr": spec.lr_head}]
        max_lrs = [spec.lr_head]
    else:
        groups = [
            {"params": head_parameters(model, name), "lr": spec.lr_head},
            {"params": backbone_parameters(model, name), "lr": spec.lr_backbone},
        ]
        max_lrs = [spec.lr_head, spec.lr_backbone]
    optimizer = torch.optim.AdamW(groups)
    steps_per_epoch = len(train_loader)
    # The schedule spans the full max-epoch budget; early stopping simply cuts it.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, total_steps=epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    state = RunState()
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        state = RunState(**ck["state"])
        _set_rng_state(ck["rng"])
        print(f"[{name}] resumed after epoch {state.epoch}", flush=True)
    elif history_path.exists():
        history_path.rename(history_path.with_suffix(".csv.bak"))
    if not history_path.exists():
        with history_path.open("w", newline="") as f:
            csv.writer(f).writerow(HISTORY_COLUMNS)

    while state.epoch < epochs and state.bad_epochs < EARLY_STOP_PATIENCE:
        epoch = state.epoch + 1
        frozen = spec.pretrained and epoch == 1
        set_backbone_frozen(model, name, frozen)
        train_mode(model, name, frozen)
        loss_sum, correct, seen = 0.0, 0, 0
        synchronize(device)
        t0 = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += int((out.argmax(1) == y).sum())
            seen += len(y)
        synchronize(device)
        seconds = time.perf_counter() - t0

        val_loss, val_acc, val_f1, *_ = evaluate_loader(model, val_loader, device)
        state.epoch = epoch
        if val_f1 > state.best_f1:
            state.best_f1, state.best_epoch, state.bad_epochs = val_f1, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": val_f1}, best_path)
        else:
            state.bad_epochs += 1
        row = [epoch, loss_sum / seen, correct / seen, val_loss, val_acc, val_f1, seconds]
        with history_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state": state.__dict__,
                "rng": _rng_state(),
            },
            last_path,
        )
        print(
            f"[{name}] ep {epoch}/{epochs} train_loss={row[1]:.4f} train_acc={row[2]:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} sec={seconds:.1f}",
            flush=True,
        )
    return pd.read_csv(history_path)
