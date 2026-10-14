"""Dataset access: paths, annotations, labels and the official split."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data" / "WaterMeters"
DIGITS_DIR: Path = ROOT / "digits"
SPLITS_CSV: Path = ROOT / "splits.csv"

#: sha256 of the professor's splits.csv (github.com/eron93br/vc-20262-p2).
SPLITS_SHA256 = "ca4c5d8ad131d63eae3a8acd5bd6ac1091f55e02426d8ad083af98cfa65abf2b"

N_INT_DIGITS = 5
N_DEC_DIGITS = 3
N_DIGITS = N_INT_DIGITS + N_DEC_DIGITS
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

_NAME_RE = re.compile(r"^id_(?P<id>\d+)_value_(?P<int>\d+)_(?P<dec>\d+)\.jpg$")

FloatArray = npt.NDArray[np.float32]


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_from_name(photo_name: str) -> str | None:
    """Parse the 8-char reading label from a photo filename.

    The integer part is left-padded to 5 digits and the decimal part right-padded
    to 3. Returns None when the reading does not fit the 5+3 format.
    """
    m = _NAME_RE.match(photo_name)
    if m is None:
        return None
    integer, decimal = m.group("int"), m.group("dec")
    if len(integer) > N_INT_DIGITS or len(decimal) > N_DEC_DIGITS:
        return None
    return integer.zfill(N_INT_DIGITS) + decimal.ljust(N_DEC_DIGITS, "0")


def label_from_value(value: float) -> str | None:
    """Label from the float `value` column (decimals rounded to 3, then 5+3 format)."""
    text = f"{value:.3f}"
    integer, decimal = text.split(".")
    if len(integer) > N_INT_DIGITS:
        return None
    return integer.zfill(N_INT_DIGITS) + decimal


def image_id_from_name(photo_name: str) -> int:
    m = _NAME_RE.match(photo_name)
    if m is None:
        raise ValueError(f"unexpected photo name: {photo_name}")
    return int(m.group("id"))


def load_annotations() -> pd.DataFrame:
    """data.csv plus: image_id, label (None if invalid) and split (NaN if not in splits.csv)."""
    df = pd.read_csv(DATA_DIR / "data.csv")
    df["image_id"] = df["photo_name"].map(image_id_from_name)
    df["label"] = df["photo_name"].map(label_from_name)
    splits = pd.read_csv(SPLITS_CSV)
    return df.merge(splits, on="photo_name", how="left")


def valid_annotations() -> pd.DataFrame:
    """Rows with a valid 5+3 label that appear in the official splits (1243 expected)."""
    df = load_annotations()
    return df[df["label"].notna() & df["split"].notna()].reset_index(drop=True)


def polygon_pixels(location: str, width: int, height: int) -> FloatArray:
    """Parse the `location` column (normalized polygon) into pixel coordinates (N, 2)."""
    points = ast.literal_eval(location)["data"]
    return np.array([[p["x"] * width, p["y"] * height] for p in points], dtype=np.float32)


def read_image(photo_name: str) -> npt.NDArray[np.uint8]:
    """Read a full photo as BGR uint8; raises if it cannot be decoded."""
    img = cv2.imread(str(DATA_DIR / "images" / photo_name))
    if img is None:
        raise OSError(f"cannot decode {photo_name}")
    return np.asarray(img, dtype=np.uint8)


@dataclass(frozen=True)
class IntegrityReport:
    """Result of the download integrity check (F1 gate)."""

    n_rows: int
    n_images: int
    n_masks: int
    n_collage: int
    n_unreadable_images: int
    n_unreadable_masks: int
    missing_images: int
    missing_masks: int
    splits_sha256: str
    splits_ok: bool

    @property
    def ok(self) -> bool:
        return (
            self.n_rows == self.n_images == self.n_masks == self.n_collage
            and self.n_unreadable_images == 0
            and self.n_unreadable_masks == 0
            and self.missing_images == 0
            and self.missing_masks == 0
            and self.splits_ok
        )


def check_integrity() -> IntegrityReport:
    """Verify file counts against data.csv, that every image/mask decodes, and splits.csv."""
    df = pd.read_csv(DATA_DIR / "data.csv")
    names = set(df["photo_name"])
    counts = {k: sorted(p.name for p in (DATA_DIR / k).glob("*.jpg")) for k in ("images", "masks", "collage")}
    bad_img = sum(cv2.imread(str(DATA_DIR / "images" / n)) is None for n in names)
    bad_mask = sum(cv2.imread(str(DATA_DIR / "masks" / n), cv2.IMREAD_GRAYSCALE) is None for n in names)
    sha = sha256_of(SPLITS_CSV)
    return IntegrityReport(
        n_rows=len(df),
        n_images=len(counts["images"]),
        n_masks=len(counts["masks"]),
        n_collage=len(counts["collage"]),
        n_unreadable_images=bad_img,
        n_unreadable_masks=bad_mask,
        missing_images=len(names - set(counts["images"])),
        missing_masks=len(names - set(counts["masks"])),
        splits_sha256=sha,
        splits_ok=sha == SPLITS_SHA256,
    )
