"""Extra challenge: locate digit boundaries on the strip instead of cutting 8 equal 48 px cells.

Idea: the wheels sit on a regular grid, but the grid of a given photo does not start exactly at
x=0 and its pitch is not always 48 px (the rectified polygon is only approximately the display).
Between two wheels (and at the margins) the columns have almost no vertical variation, whereas
digit strokes make the columns vary along the vertical axis. So the boundary signal is the
smoothed column standard deviation of the grayscale strip; the best grid (offset, pitch) is the
one whose 9 boundaries sit on the lowest values of that signal, and each interior boundary is then
refined locally. Every cell is resampled to 48x64, so the models and the protocol are unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

from hidrometros.preprocessing import CROP_SIZE, N_CELLS, STRIP_SIZE

Strip = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

STRIP_W: int = STRIP_SIZE[0]
DEFAULT_PITCH: float = STRIP_W / N_CELLS
PITCH_RANGE: tuple[float, float] = (42.0, 54.0)
OFFSET_RANGE: tuple[int, int] = (-6, 14)
REFINE_RADIUS = 5
#: Penalty (in units of boundary signal) per pixel of deviation from the equal-width grid.
PRIOR_WEIGHT = 0.004
#: The grid must reach close to the right end of the strip, otherwise the last wheel is cut.
MIN_RIGHT_EDGE = 372.0
RIGHT_EDGE_WEIGHT = 0.05


def boundary_signal(strip: Strip) -> FloatArray:
    """Smoothed per-column vertical variation of the grayscale strip, normalized to [0, 1]."""
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64)
    col = gray.std(axis=0)
    kernel = np.ones(5) / 5.0
    col = np.convolve(np.pad(col, 2, mode="edge"), kernel, mode="valid")
    span = float(col.max() - col.min())
    return (col - col.min()) / (span if span > 0 else 1.0)


def _sample(signal: FloatArray, x: float) -> float:
    return float(signal[int(np.clip(round(x), 0, STRIP_W - 1))])


def find_boundaries(strip: Strip) -> list[float]:
    """Return the 9 cell boundaries (x positions, left edge of cell 0 to right edge of cell 7)."""
    sig = boundary_signal(strip)
    best_cost, best = float("inf"), (0.0, DEFAULT_PITCH)
    for pitch in np.arange(PITCH_RANGE[0], PITCH_RANGE[1] + 0.01, 0.5):
        for off in range(OFFSET_RANGE[0], OFFSET_RANGE[1] + 1):
            if off + pitch * N_CELLS > STRIP_W + 10:
                continue
            xs = [off + k * pitch for k in range(1, N_CELLS)]  # interior boundaries
            cost = float(np.mean([_sample(sig, x) for x in xs]))
            cost += PRIOR_WEIGHT * (abs(pitch - DEFAULT_PITCH) * N_CELLS / 2 + abs(off))
            cost += RIGHT_EDGE_WEIGHT * max(0.0, MIN_RIGHT_EDGE - (off + pitch * N_CELLS))
            if cost < best_cost:
                best_cost, best = cost, (float(off), float(pitch))
    best_off, best_pitch = best
    bounds = [best_off + k * best_pitch for k in range(N_CELLS + 1)]
    refined = [bounds[0]]
    for k in range(1, N_CELLS):
        lo = max(bounds[k] - REFINE_RADIUS, refined[-1] + PITCH_RANGE[0] - 4)
        hi = bounds[k] + REFINE_RADIUS
        cands = np.arange(lo, hi + 0.5, 1.0)
        if len(cands) == 0:
            refined.append(bounds[k])
            continue
        costs = [_sample(sig, x) + 0.003 * abs(x - bounds[k]) for x in cands]
        refined.append(float(cands[int(np.argmin(costs))]))
    refined.append(bounds[-1])
    return refined


def split_strip(strip: Strip) -> list[Image.Image]:
    """Cut the strip at the located boundaries; every cell is resampled to 48x64 RGB."""
    bounds = find_boundaries(strip)
    rgb = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
    crops: list[Image.Image] = []
    for k in range(N_CELLS):
        a = int(np.clip(round(bounds[k]), 0, STRIP_W - 8))
        b = int(np.clip(round(bounds[k + 1]), a + 8, STRIP_W))
        cell = rgb[:, a:b]
        crops.append(Image.fromarray(cell).resize(CROP_SIZE, Image.Resampling.BILINEAR))
    return crops


def edge_ink_fraction(crop: Image.Image, width: int = 3) -> tuple[bool, bool]:
    """Whether a crop has digit ink touching its left / right border (a digit cut by the split).

    Ink = column vertical variation above 60% of the crop's own median over its central columns.
    """
    gray = np.asarray(crop.convert("L"), dtype=np.float64)
    col = gray.std(axis=0)
    centre = float(np.median(col[CROP_SIZE[0] // 4 : 3 * CROP_SIZE[0] // 4]))
    thr = 0.6 * centre
    return bool(col[:width].mean() > thr), bool(col[-width:].mean() > thr)
