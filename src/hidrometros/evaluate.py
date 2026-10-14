"""Test-split evaluation: metrics, cost measurements, reports and results.csv."""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import classification_report

from hidrometros.metrics import classification_metrics, reading_accuracy
from hidrometros.models import MODEL_NAMES, PROTOCOL, build_model, count_params_m
from hidrometros.reproducibility import set_seed
from hidrometros.train import (
    BASE,
    Variant,
    batch_size_for,
    eval_transform,
    evaluate_loader,
    get_device,
    load_crops,
    make_loader,
    split_index,
    synchronize,
)

RESULT_COLUMNS = [
    "model", "acc_digit", "f1_macro", "acc_reading", "params_M",
    "s_per_epoch", "ms_per_image", "ckpt_MB", "best_epoch",
]  # fmt: skip
WARMUP_RUNS = 20
TIMED_RUNS = 200


def load_best(name: str, variant: Variant, device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load the best-by-val-F1 checkpoint; returns the model and its epoch."""
    path = variant.ckpt_root / name / "best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(name, pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), int(ck["epoch"])


@torch.inference_mode()
def ms_per_image(model: torch.nn.Module, name: str, device: torch.device, sample: torch.Tensor) -> float:
    """Median latency (ms) of batch-1 forward passes: 20 warm-up, then 200 timed, synced each run.

    The input is an already transformed tensor on the device, so transform cost is excluded.
    """
    model.eval()
    x = eval_transform(PROTOCOL[name])(sample).unsqueeze(0).to(device)
    for _ in range(WARMUP_RUNS):
        model(x)
    synchronize(device)
    times: list[float] = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        model(x)
        synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def evaluate_model(name: str, variant: Variant = BASE) -> tuple[dict[str, float | str], pd.DataFrame, npt.NDArray[np.int64]]:
    """Evaluate one model on the TEST split only. Returns the results row, predictions and confusion matrix."""
    set_seed(42)
    device = get_device()
    spec = PROTOCOL[name]
    model, best_epoch = load_best(name, variant, device)
    test_idx = split_index("test", variant.index_csv)
    images = load_crops(test_idx)
    loader = make_loader(test_idx, spec, batch_size_for(spec, device), train=False, images=images)
    _, _, _, y_true, y_pred, conf = evaluate_loader(model, loader, device)
    m = classification_metrics(y_true, y_pred)
    reading = reading_accuracy(test_idx["image_id"].to_numpy(dtype=np.int64), y_true, y_pred)
    hist = pd.read_csv(variant.history(name))
    ckpt = variant.ckpt_root / name / "best.pt"
    row: dict[str, float | str] = {
        "model": name,
        "acc_digit": float(m["acc_digit"]),
        "f1_macro": float(m["f1_macro"]),
        "acc_reading": reading,
        "params_M": count_params_m(model),
        "s_per_epoch": float(hist["seconds"].median()),
        "ms_per_image": ms_per_image(model, name, device, images[0]),
        "ckpt_MB": ckpt.stat().st_size / 1e6,
        "best_epoch": best_epoch,
    }
    preds = test_idx[["photo_name", "image_id", "position"]].copy()
    preds["y_true"], preds["y_pred"], preds["confidence"] = y_true, y_pred, conf
    variant.reports_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(variant.reports_dir / f"predictions_{name}.csv", index=False, float_format="%.5f")
    (variant.reports_dir / f"classification_report_{name}.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(10)), digits=4, zero_division=0)
    )
    cm = np.asarray(m["confusion_matrix"], dtype=np.int64)
    return row, preds, cm


def format_row(row: dict[str, float | str]) -> dict[str, str]:
    """Fixed-precision strings, so results.csv and the README table are literally the same text."""
    return {
        "model": str(row["model"]),
        "acc_digit": f"{float(row['acc_digit']):.4f}",
        "f1_macro": f"{float(row['f1_macro']):.4f}",
        "acc_reading": f"{float(row['acc_reading']):.4f}",
        "params_M": f"{float(row['params_M']):.2f}",
        "s_per_epoch": f"{float(row['s_per_epoch']):.1f}",
        "ms_per_image": f"{float(row['ms_per_image']):.2f}",
        "ckpt_MB": f"{float(row['ckpt_MB']):.1f}",
        "best_epoch": str(int(row["best_epoch"])),
    }


def write_results(variant: Variant = BASE, names: tuple[str, ...] = MODEL_NAMES) -> pd.DataFrame:
    """Evaluate `names` on test and write the results CSV (and confusion matrices as .npy).

    A subset of the models replaces only its own rows and matrices and keeps the others as they are, so
    re-running one model does not re-measure (or change) the cost columns of the rest.
    """
    cm_path = variant.reports_dir / "confusion_matrices.json"
    rows: dict[str, dict[str, str]] = {}
    cms: dict[str, list[list[int]]] = {}
    if set(names) != set(MODEL_NAMES):
        old = pd.read_csv(variant.results_csv, dtype=str)
        rows = {str(r["model"]): {str(k): str(v) for k, v in r.items()} for r in old.to_dict("records")}
        cms = json.loads(cm_path.read_text())
    for name in names:
        row, _, cm = evaluate_model(name, variant)
        rows[name] = format_row(row)
        cms[name] = cm.tolist()
    df = pd.DataFrame([rows[n] for n in MODEL_NAMES], columns=RESULT_COLUMNS)
    df.to_csv(variant.results_csv, index=False)
    cm_path.write_text(json.dumps({n: cms[n] for n in MODEL_NAMES}))
    return df
