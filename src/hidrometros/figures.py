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
LABEL_OFFSETS = {"smallcnn": (10, -4), "alexnet": (14, -30), "vgg16_bn": (0, 52), "resnet18": (-14, 12)}
LABEL_HA = {"smallcnn": "left", "alexnet": "left", "vgg16_bn": "center", "resnet18": "right"}
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
        ax.set_xlabel("dígito previsto")
        ax.set_ylabel("dígito verdadeiro")
        ax.set_title(name)
    fig.colorbar(im, ax=axes, shrink=0.6, label="fração da classe verdadeira (normalizada por linha)")
    fig.suptitle("Matrizes de confusão no teste (normalizadas pela classe verdadeira)")
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
    ax.set_xlabel("posição do dígito (da esquerda para a direita; int = rodas inteiras, dec = rodas decimais)")
    ax.set_ylabel("acurácia por recorte (teste)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Acurácia por posição do dígito (conjunto de teste)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="modelo")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def training_curves(variant: Variant = BASE, out: Path = FIG_DIR / "training_curves.png") -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=False, constrained_layout=True)
    for j, name in enumerate(MODEL_NAMES):
        h = pd.read_csv(variant.history(name))
        for i, (tr, va, label) in enumerate([("train_loss", "val_loss", "perda"), ("train_acc", "val_acc", "acurácia")]):
            ax = axes[i, j]
            ax.plot(h["epoch"], h[tr], "-o", color=COLORS[name], label="treino")
            ax.plot(h["epoch"], h[va], "--s", color=COLORS[name], alpha=0.8, label="validação")
            ax.set_xticks(h["epoch"])
            ax.set_xlabel("época")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name)
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle("Curvas de treino: linha cheia = treino (com aumento de dados), tracejada = validação")
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
        ax.annotate(f"{name}\n{params:.1f}M parâmetros", (ms, f1), textcoords="offset points", xytext=LABEL_OFFSETS[name], fontsize=9, ha=LABEL_HA[name])
    ax.set_xlabel("tempo de inferência por imagem, batch 1 (ms, escala log; menor é melhor)")
    ax.set_ylabel("F1-macro no teste (maior é melhor)")
    ax.set_xscale("log")
    ax.set_title("Desempenho × custo (área do marcador ~ parâmetros)")
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
