"""Strip-to-digit splitting. Original scaffold by Mikael de Albuquerque Mulatinho."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

STRIP_SIZE: tuple[int, int] = (384, 64)  # (width, height)
CROP_SIZE: tuple[int, int] = (48, 64)
N_CELLS = 8


def make_digit_crops(strip: Image.Image, expected_size: tuple[int, int] = STRIP_SIZE) -> list[Image.Image]:
    """Split an already oriented 384x64 strip into 8 equal-width 48x64 crops, left to right.

    This does NOT locate the strip, correct perspective or decide orientation (see
    `hidrometros.prep`). Equal widths assume digit cells sit exactly on a 48 px grid, which
    is the source of the errors discussed in questions 3.2 and 3.3.
    """
    strip = strip.convert("RGB")
    if strip.size != expected_size:
        raise ValueError(f"expected size {expected_size} (WxH), got {strip.size}")
    width, height = strip.size
    crop_width = width // N_CELLS
    return [strip.crop((i * crop_width, 0, (i + 1) * crop_width, height)) for i in range(N_CELLS)]


def save_crops(crops: Sequence[Image.Image], output_dir: Path, stem: str = "reading") -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, crop in enumerate(crops):
        path = output_dir / f"{stem}_digit_{i}.png"
        crop.save(path)
        paths.append(path)
    return paths
