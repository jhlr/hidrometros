"""Display rectification, orientation rules and 8-digit crop generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image

from hidrometros import data
from hidrometros.preprocessing import CROP_SIZE, STRIP_SIZE, make_digit_crops

Strip = npt.NDArray[np.uint8]  # BGR, shape (64, 384, 3)

LOCAL_DIR: Path = data.ROOT / "local"
STRIPS_NPZ: Path = LOCAL_DIR / "strips.npz"
ORIENTATION_CSV: Path = LOCAL_DIR / "orientation.csv"
PREP_STATS_JSON: Path = data.ROOT / "reports" / "prep_stats.json"
OVERRIDES_CSV: Path = data.ROOT / "data_overrides.csv"

#: A rectangle whose long axis is farther than this from horizontal counts as a rotated display.
ROTATION_TOLERANCE_DEG = 30.0
#: Red-wheel rule: red mass of one quarter must be at least this multiple of the other.
RED_RATIO = 2.0
#: Strips with fewer red pixels than this carry no orientation evidence from the rule.
MIN_RED_PIXELS = 200


@dataclass(frozen=True)
class Rectified:
    strip: Strip
    long_axis_deg: float  # angle of the rectangle long side in [0, 180)
    aspect: float


def rectify(photo_name: str, location: str) -> Rectified:
    """Warp the annotated display into a 384x64 horizontal strip.

    Uses `cv2.minAreaRect` on the pixel polygon, so vertical and tilted displays come out
    horizontal. A 180 degree ambiguity remains and is resolved by `red_orientation` and the
    orientation classifier.
    """
    img = data.read_image(photo_name)
    h, w = img.shape[:2]
    pts = data.polygon_pixels(location, w, h)
    rect = cv2.minAreaRect(pts)
    (rw, rh) = rect[1]
    box = cv2.boxPoints(rect)
    edges = [float(np.linalg.norm(box[(i + 1) % 4] - box[i])) for i in range(4)]
    i = int(np.argmax(edges))
    a, b, c, d = box[i], box[(i + 1) % 4], box[(i + 2) % 4], box[(i + 3) % 4]
    if b[0] < a[0]:  # long edge a->b must point right
        a, b, c, d = b, a, d, c
    if a[1] > d[1]:  # a is the top-left corner
        a, b, c, d = d, c, b, a
    src = np.array([a, b, c, d], np.float32)
    sw, sh = STRIP_SIZE
    dst = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    strip = np.asarray(cv2.warpPerspective(img, m, STRIP_SIZE), dtype=np.uint8)
    vec = box[(i + 1) % 4] - box[i]
    theta = float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)
    return Rectified(strip, theta, float(max(rw, rh) / max(min(rw, rh), 1e-6)))


def rotated_by_geometry(long_axis_deg: float) -> bool:
    """True when the display long axis was not within 30 degrees of horizontal."""
    return ROTATION_TOLERANCE_DEG <= long_axis_deg <= 180.0 - ROTATION_TOLERANCE_DEG


def red_mass(strip: Strip) -> npt.NDArray[np.float32]:
    """Binary red-pixel map (HSV: h<8 or h>170, s>90, v>60)."""
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h < 8) | (h > 170)) & (s > 90) & (v > 60)).astype(np.float32)


def red_orientation(strip: Strip) -> str:
    """Red-wheel rule: decimal wheels are red and sit on the right of an upright strip.

    Returns one of `upright`, `upside_down`, `ambiguous`, `no_red`.
    """
    red = red_mass(strip)
    quarter = STRIP_SIZE[0] // 4
    left = float(red[:, :quarter].sum())
    right = float(red[:, -quarter:].sum())
    if red.sum() < MIN_RED_PIXELS:
        return "no_red"
    if right >= RED_RATIO * left:
        return "upright"
    if left >= RED_RATIO * right:
        return "upside_down"
    return "ambiguous"


def rotate180(strip: Strip) -> Strip:
    return np.ascontiguousarray(strip[::-1, ::-1])


def build_strips(df: pd.DataFrame) -> tuple[dict[str, Strip], pd.DataFrame]:
    """Rectify every valid image. Returns strips by photo_name and a geometry table."""
    strips: dict[str, Strip] = {}
    rows: list[dict[str, object]] = []
    for name, loc in zip(df["photo_name"], df["location"], strict=True):
        r = rectify(name, loc)
        strips[name] = r.strip
        rows.append(
            {
                "photo_name": name,
                "long_axis_deg": r.long_axis_deg,
                "aspect": r.aspect,
                "rotated_geometry": rotated_by_geometry(r.long_axis_deg),
                "red_rule": red_orientation(r.strip),
            }
        )
    return strips, pd.DataFrame(rows)


def load_overrides() -> dict[str, bool]:
    """Manual audit corrections (photo_name -> flip 180 degrees), applied last."""
    if not OVERRIDES_CSV.exists():
        return {}
    df = pd.read_csv(OVERRIDES_CSV)
    return {str(n): bool(f) for n, f in zip(df["photo_name"], df["flip"], strict=True)}


def crop_strip(strip: Strip) -> list[Image.Image]:
    """Equal-width split of an oriented strip into 8 crops of 48x64."""
    rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    return make_digit_crops(Image.fromarray(rgb))


def write_crops(
    strips: dict[str, Strip],
    df: pd.DataFrame,
    out_dir: Path,
    splitter: Callable[[Strip], list[Image.Image]] = crop_strip,
) -> pd.DataFrame:
    """Save `digits/{split}/{digit}/{image_id}_{position}.png` and return the index table."""
    rows: list[dict[str, object]] = []
    for name, image_id, label, split in zip(
        df["photo_name"], df["image_id"], df["label"], df["split"], strict=True
    ):
        crops = splitter(strips[name])
        for pos, crop in enumerate(crops):
            digit = int(label[pos])
            assert crop.size == CROP_SIZE
            path = out_dir / split / str(digit) / f"{image_id}_{pos}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path)
            rows.append(
                {
                    "image_id": image_id,
                    "photo_name": name,
                    "position": pos,
                    "digit": digit,
                    "split": split,
                    "crop_path": str(path.relative_to(data.ROOT)),
                }
            )
    return pd.DataFrame(rows)


def save_strips(strips: dict[str, Strip]) -> None:
    """Persist oriented strips locally (gitignored) as one array plus the name list."""
    STRIPS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    names = list(strips)
    np.savez_compressed(STRIPS_NPZ, names=np.array(names), strips=np.stack([strips[n] for n in names]))


def load_strips() -> dict[str, Strip]:
    """Inverse of `save_strips`."""
    z = np.load(STRIPS_NPZ)
    return {str(n): np.asarray(s, dtype=np.uint8) for n, s in zip(z["names"], z["strips"], strict=True)}


def stats_to_json(stats: dict[str, object], path: Path = PREP_STATS_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, default=str))
