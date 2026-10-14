"""Generated parts of README.md (results table, protocol table), always derived from files on disk."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from hidrometros import data
from hidrometros.criteria import RESULT_COLUMNS
from hidrometros.models import MODEL_NAMES, PROTOCOL
from hidrometros.train import BASE

README: Path = data.ROOT / "README.md"


def results_table(results_csv: Path = BASE.results_csv) -> str:
    """Markdown table that repeats the strings of results.csv verbatim."""
    df = pd.read_csv(results_csv, dtype=str)
    lines = ["| " + " | ".join(RESULT_COLUMNS) + " |", "|" + "---|" * len(RESULT_COLUMNS)]
    lines += ["| " + " | ".join(row) + " |" for row in df.values.tolist()]
    return "\n".join(lines)


def protocol_table(results_csv: Path = BASE.results_csv) -> str:
    """Protocol table with the real batch sizes used (MPS/CPU column) and best epochs."""
    best = pd.read_csv(results_csv, dtype=str).set_index("model")["best_epoch"]
    lines = [
        "| Modelo | Pesos | Entrada | Batch (usado: MPS) | LR cabeça | LR backbone | Épocas máx. | Melhor época |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in MODEL_NAMES:
        s = PROTOCOL[name]
        weights = "ImageNet (IMAGENET1K_V1)" if s.pretrained else "do zero"
        bb = "—" if s.lr_backbone is None else f"{s.lr_backbone:g}"
        lines.append(
            f"| {name} | {weights} | {s.input_hw[0]}×{s.input_hw[1]} | {s.batch_mps_cpu} | {s.lr_head:g} | {bb} | {s.max_epochs} | {best[name]} |"
        )
    return "\n".join(lines)


def replace_block(text: str, tag: str, body: str) -> str:
    """Replace the content between `<!-- tag:start -->` and `<!-- tag:end -->`."""
    pattern = re.compile(rf"(<!-- {tag}:start -->\n).*?(\n<!-- {tag}:end -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"marker {tag} not found in README")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> None:
    text = README.read_text()
    text = replace_block(text, "results", results_table())
    text = replace_block(text, "protocol", protocol_table())
    README.write_text(text)
    print("README generated blocks updated")


if __name__ == "__main__":
    main()
