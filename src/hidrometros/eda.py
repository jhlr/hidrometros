"""Exploratory statistics that back the answers to questions 1.1, 1.2, 2.1 and 2.2."""

from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from hidrometros import data


def mask_polygon_agreement() -> dict[str, float]:
    """IoU between the binarized mask image and the rasterized polygon (all 1244 rows)."""
    df = pd.read_csv(data.DATA_DIR / "data.csv")
    ious: list[float] = []
    for name, loc in zip(df["photo_name"], df["location"], strict=True):
        mask = cv2.imread(str(data.DATA_DIR / "masks" / name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise OSError(name)
        h, w = mask.shape
        poly = np.zeros((h, w), np.uint8)
        pts = data.polygon_pixels(loc, w, h).round().astype(np.int32)
        cv2.fillPoly(poly, [pts], 1)
        m = mask > 127
        union = np.logical_or(m, poly > 0).sum()
        ious.append(float(np.logical_and(m, poly > 0).sum() / union) if union else 1.0)
    arr = np.array(ious)
    return {"iou_median": float(np.median(arr)), "iou_p05": float(np.percentile(arr, 5)), "iou_min": float(arr.min())}


def summarize() -> dict[str, object]:
    df = data.load_annotations()
    valid = df[df["label"].notna() & df["split"].notna()]
    digits = Counter("".join(valid["label"]))
    total = sum(digits.values())
    per_pos = [Counter(lbl[p] for lbl in valid["label"]) for p in range(data.N_DIGITS)]
    value_name_agree = int(
        (df["value"].map(data.label_from_value) == df["label"])[df["label"].notna()].sum()
    )
    sizes: Counter[str] = Counter()
    for n in df["photo_name"]:
        img = cv2.imread(str(data.DATA_DIR / "images" / n))
        assert img is not None
        sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1
    return {
        "columns_data_csv": list(pd.read_csv(data.DATA_DIR / "data.csv").columns),
        "n_rows": len(df),
        "n_valid": len(valid),
        "discarded": df[df["label"].isna()]["photo_name"].tolist(),
        "value_vs_filename_agree_on_valid": value_name_agree,
        "split_counts_images": valid["split"].value_counts().to_dict(),
        "digit_share": {d: round(digits[d] / total, 4) for d in sorted(digits)},
        "digit_counts_per_position": [{d: c[d] for d in sorted(c)} for c in per_pos],
        "n_crops_expected": len(valid) * data.N_DIGITS,
        "image_sizes": dict(sizes.most_common(6)),
        "mask_polygon_agreement": mask_polygon_agreement(),
        "polygon_points": pd.read_csv(data.DATA_DIR / "data.csv")["location"]
        .map(lambda s: len(data.polygon_pixels(s, 1, 1)))
        .value_counts()
        .to_dict(),
        "value_range": [float(df["value"].min()), float(df["value"].max())],
    }


def main() -> None:
    print(json.dumps(summarize(), indent=2, default=str))


if __name__ == "__main__":
    main()
