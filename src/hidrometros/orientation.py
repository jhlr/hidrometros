"""Self-supervised 0 vs 180 degree orientation classifier for display strips.

Positives are train-split strips that the red-wheel rule judged upright; negatives are the same
strips rotated by 180 degrees. Nothing from val/test is used to fit or tune it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F

from hidrometros.prep import Strip, rotate180
from hidrometros.reproducibility import set_seed

H, W = 32, 192  # strips are downsampled from 64x384


def device() -> torch.device:
    """CPU on purpose: the model is tiny and CPU keeps the orientation decisions reproducible."""
    return torch.device("cpu")


def to_tensor(strips: list[Strip]) -> torch.Tensor:
    """Grayscale, downsampled, per-image standardized (removes most of the red-wheel tone cue)."""
    out = np.stack([_gray(s) for s in strips])[:, None]
    return torch.from_numpy(out)


def _gray(strip: Strip) -> npt.NDArray[np.float32]:
    import cv2

    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    return ((g - g.mean()) / (g.std() + 1e-6)).astype(np.float32)


class OrientNet(nn.Module):
    """Tiny CNN; output logit > 0 means upright."""

    def __init__(self) -> None:
        super().__init__()
        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Linear(64 * 2 * 12, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return torch.as_tensor(self.head(z).squeeze(1))


def _augment(x: torch.Tensor, generator: torch.Generator, mask_right: bool) -> torch.Tensor:
    """Random gain/bias, small horizontal shift, optional masking of the red (right) region."""
    b = x.shape[0]
    gain = 0.7 + 0.6 * torch.rand(b, 1, 1, 1, generator=generator, device=x.device)
    bias = 0.3 * torch.randn(b, 1, 1, 1, generator=generator, device=x.device)
    x = x * gain + bias
    shift = int(torch.randint(-6, 7, (1,), generator=generator, device=x.device))
    x = torch.roll(x, shifts=shift, dims=3)
    if mask_right:
        drop = torch.rand(b, 1, 1, 1, generator=generator, device=x.device) < 0.5
        cut = x.clone()
        cut[..., W * 5 // 8 :] = 0.0
        x = torch.where(drop, cut, x)
    return x


def train_classifier(
    upright: list[Strip], epochs: int = 25, seed: int = 42, holdout_frac: float = 0.2
) -> tuple[OrientNet, dict[str, float]]:
    """Fit on upright strips (+ their 180 rotations). Returns model and held-out train metrics.

    Held-out accuracy is measured on a split of the TRAIN strips only: once on the full strip and
    once with the right (red) 3/8 zeroed, to show how much it relies on the red cue.
    """
    set_seed(seed)
    dev = device()
    n = len(upright)
    perm = np.random.default_rng(seed).permutation(n)
    n_hold = int(n * holdout_frac)
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    def pairs(idx: npt.NDArray[np.int64]) -> tuple[torch.Tensor, torch.Tensor]:
        pos = [upright[i] for i in idx]
        neg = [rotate180(s) for s in pos]
        x = to_tensor(pos + neg)
        y = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        return x, y

    xf, yf = pairs(fit_idx)
    xh, yh = pairs(hold_idx)
    model = OrientNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    gen = torch.Generator(device=dev).manual_seed(seed)
    bs = 64
    steps = epochs * ((len(xf) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    xf, yf = xf.to(dev), yf.to(dev)
    order_gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xf), generator=order_gen).to(dev)
        for i in range(0, len(xf), bs):
            idx = order[i : i + bs]
            x = _augment(xf[idx], gen, mask_right=True)
            loss = F.binary_cross_entropy_with_logits(model(x), yf[idx])
            opt.zero_grad()
            torch.autograd.backward(loss)
            opt.step()
            sched.step()
    metrics = {
        "holdout_acc_full": _acc(model, xh, yh, mask_right=False),
        "holdout_acc_right_masked": _acc(model, xh, yh, mask_right=True),
        "n_fit_strips": float(len(fit_idx)),
        "n_holdout_strips": float(len(hold_idx)),
    }
    return model, metrics


@torch.inference_mode()
def _acc(model: OrientNet, x: torch.Tensor, y: torch.Tensor, mask_right: bool) -> float:
    model.eval()
    dev = next(model.parameters()).device
    xx = x.clone()
    if mask_right:
        xx[..., W * 5 // 8 :] = 0.0
    pred = (model(xx.to(dev)) > 0).float().cpu()
    return float((pred == y).float().mean())


def train_ensemble(
    upright: list[Strip], seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
) -> tuple[list[OrientNet], dict[str, float]]:
    """Train one classifier per seed (same train strips); metrics are averaged over members.

    Averaging the probabilities stabilizes the borderline decisions of single small models.
    """
    models: list[OrientNet] = []
    totals: dict[str, float] = {}
    for sd in seeds:
        m, met = train_classifier(upright, seed=sd)
        models.append(m)
        for k, v in met.items():
            totals[k] = totals.get(k, 0.0) + v / len(seeds)
    totals["n_members"] = float(len(seeds))
    return models, totals


@torch.inference_mode()
def prob_upright(models: list[OrientNet], strips: list[Strip]) -> npt.NDArray[np.float32]:
    """Mean probability over the ensemble that each strip is upright."""
    x = to_tensor(strips)
    probs = []
    for model in models:
        model.eval()
        dev = next(model.parameters()).device
        probs.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
    return np.mean(probs, axis=0).astype(np.float32)
