"""Local-only contact sheets for the orientation audit (never committed)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hidrometros import prep

AUDIT_DIR: Path = prep.LOCAL_DIR / "audit"


def _tile(strip: prep.Strip, caption: str) -> prep.Strip:
    banner = np.zeros((16, strip.shape[1], 3), np.uint8)
    cv2.putText(banner, caption, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return np.vstack([banner, strip])


def contact_sheets(names: list[str], captions: list[str], prefix: str, per_sheet: int = 24, cols: int = 2) -> list[Path]:
    """Write contact sheets of final-orientation strips with captions; returns their paths."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    npz = prep.load_strips()
    out: list[Path] = []
    for s in range(0, len(names), per_sheet):
        tiles = [_tile(npz[n], c) for n, c in zip(names[s : s + per_sheet], captions[s : s + per_sheet], strict=True)]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
        path = AUDIT_DIR / f"{prefix}_{s // per_sheet:02d}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        out.append(path)
    return out


def _fmt(label: str) -> str:
    return f"{label[:5]}.{label[5:]}"


def _captions(df: pd.DataFrame) -> list[str]:
    return [f"{i} {_fmt(str(r.label))} {r.tag} p={r.p_upright:.2f} {r.red_rule}" for i, r in enumerate(df.itertuples())]


def orientation_sheets(seed: int = 7, n_random: int = 50) -> dict[str, list[Path]]:
    """Sheets for every flipped strip and for `n_random` random non-flipped strips."""
    g = pd.read_csv(prep.ORIENTATION_CSV, dtype={"label": str})
    g["tag"] = np.where(g["flip_rule"], "rule", np.where(g["flip_clf"], "clf", "-"))
    flipped = g[g["flip_final"]].sort_values("photo_name")
    rest = g[~g["flip_final"]].sample(n=n_random, random_state=seed)
    return {
        "flipped": contact_sheets(flipped["photo_name"].tolist(), _captions(flipped), "flipped"),
        "random": contact_sheets(rest["photo_name"].tolist(), _captions(rest), "random"),
    }


def uncertain_sheets(low: float = 0.02, high: float = 0.98) -> list[Path]:
    """Sheets for strips decided by the classifier (flipped or not) with p(upright) in (low, high)."""
    g = pd.read_csv(prep.ORIENTATION_CSV, dtype={"label": str})
    g["tag"] = np.where(g["flip_final"], "FLIP", "-")
    u = g[g["red_rule"].isin(["no_red", "ambiguous"]) & g["p_upright"].between(low, high, inclusive="neither")]
    return contact_sheets(u["photo_name"].tolist(), _captions(u), "uncertain")


def main() -> None:
    print("uncertain", [str(p) for p in uncertain_sheets()])
    for k, v in orientation_sheets().items():
        print(k, [str(p) for p in v])


if __name__ == "__main__":
    main()
