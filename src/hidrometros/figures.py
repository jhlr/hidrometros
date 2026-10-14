"""The four required figures, drawn from results.csv, histories and test predictions."""

from __future__ import annotations

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hidrometros import data
from hidrometros.models import MODEL_NAMES
from hidrometros.train import BASE, Variant

plt.switch_backend("Agg")

FIG_DIR: Path = data.ROOT / "figures"
# Okabe-Ito colorblind-safe palette
COLORS = {"smallcnn": "#0072B2", "alexnet": "#E69F00", "vgg16_bn": "#009E73", "resnet18": "#D55E00"}


def confusion_matrices(variant: Variant = BASE, out: Path = FIG_DIR / "confusion_matrices.png") -> Path:
    cms = {k: np.array(v) for k, v in json.loads((variant.reports_dir / "confusion_matrices.json").read_text()).items()}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for ax, name in zip(axes.ravel(), MODEL_NAMES, strict=True):
        cm = cms[name].astype(float)
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{norm[i, j]:.2f}" if norm[i, j] >= 0.005 else "", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")  # fmt: skip
        ax.set_xticks(range(10), labels=[str(d) for d in range(10)])
        ax.set_yticks(range(10), labels=[str(d) for d in range(10)])
        ax.set_xlabel("predicted digit")
        ax.set_ylabel("true digit")
        ax.set_title(name)
    fig.colorbar(im, ax=axes, shrink=0.6, label="fraction of true class (row-normalized)")
    fig.suptitle("Test-set confusion matrices (normalized by true class)")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def accuracy_by_position(variant: Variant = BASE, out: Path = FIG_DIR / "accuracy_by_position.png") -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    width = 0.2
    lows: list[float] = []
    for k, name in enumerate(MODEL_NAMES):
        p = pd.read_csv(variant.reports_dir / f"predictions_{name}.csv")
        acc = (p["y_true"] == p["y_pred"]).groupby(p["position"]).mean()
        lows.append(float(acc.min()))
        ax.bar(acc.index.to_numpy(dtype=float) + (k - 1.5) * width, acc.to_numpy(dtype=float), width, label=name, color=COLORS[name])
    ax.set_xticks(range(8), labels=[f"{i}\n({'int' if i < 5 else 'dec'})" for i in range(8)])
    ax.set_xlabel("digit position (left to right; int = integer wheels, dec = decimal wheels)")
    ax.set_ylabel("test accuracy per crop")
    ax.set_ylim(max(0.0, min(lows) - 0.05), 1.0)
    ax.set_title("Accuracy by digit position (test split)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="lower left")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def training_curves(variant: Variant = BASE, out: Path = FIG_DIR / "training_curves.png") -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=False, constrained_layout=True)
    for j, name in enumerate(MODEL_NAMES):
        h = pd.read_csv(variant.history(name))
        for i, (tr, va, label) in enumerate([("train_loss", "val_loss", "loss"), ("train_acc", "val_acc", "accuracy")]):
            ax = axes[i, j]
            ax.plot(h["epoch"], h[tr], "-o", color=COLORS[name], label="train")
            ax.plot(h["epoch"], h[va], "--s", color=COLORS[name], alpha=0.8, label="val")
            ax.set_xticks(h["epoch"])
            ax.set_xlabel("epoch")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name)
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle("Training curves: solid = train (augmented), dashed = validation")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def performance_vs_cost(results_csv: Path = BASE.results_csv, out: Path = FIG_DIR / "performance_vs_cost.png") -> Path:
    df = pd.read_csv(results_csv)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i in range(len(df)):
        name = str(df["model"].iloc[i])
        ms, f1, params = float(df["ms_per_image"].iloc[i]), float(df["f1_macro"].iloc[i]), float(df["params_M"].iloc[i])
        ax.scatter([ms], [f1], s=60 + 12 * params, color=COLORS[name], edgecolor="black", zorder=3)
        ax.annotate(f"{name}\n{params:.1f}M params", (ms, f1), textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel("inference time per image, batch 1 (ms, log scale; lower is better)")
    ax.set_ylabel("test F1-macro (higher is better)")
    ax.set_xscale("log")
    ax.set_title("Performance vs cost (marker area ~ parameters)")
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.25, y=0.15)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    for fn in (confusion_matrices, accuracy_by_position, training_curves, performance_vs_cost):
        print(fn())


if __name__ == "__main__":
    main()
