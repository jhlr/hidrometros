"""Guard for the smallcnn defect (2026-10-14): global average pooling discarded where a stroke is.

Digit recognition depends on the arrangement of strokes. With a 3x3-conv, two-pool trunk the receptive
field is 18 px on a 64x48 crop, so a head that averages the whole map over space only sees a bag of
local features and the model stayed at the majority class (test acc 0.2246).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hidrometros.models import NUM_CLASSES, PROTOCOL, SmallCNN

H, W = PROTOCOL["smallcnn"].input_hw


def _batch(n: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Class k is the cell of a 5x2 grid holding a bright 6x6 square (jittered by up to 2 px)."""
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    x = torch.randn(n, 3, H, W, generator=g) * 0.1
    jitter = torch.randint(-2, 3, (n, 2), generator=g)
    for i, k in enumerate(y.tolist()):
        row, col = divmod(k, 2)
        top, left = row * 12 + 3 + int(jitter[i, 0]), col * 24 + 9 + int(jitter[i, 1])
        x[i, :, top : top + 6, left : left + 6] += 2.0
    return x, y


def test_smallcnn_uses_spatial_layout() -> None:
    """Given a task whose label is only WHERE a square is, When SmallCNN trains 150 steps, Then acc >= 0.9.

    A head that pools over all positions cannot see location and stays near 0.4 (border effects only).
    """
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = SmallCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(150):
        x, y = _batch(64, g)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    x, y = _batch(512, g)
    with torch.inference_mode():
        acc = float((model(x).argmax(1) == y).float().mean())
    assert acc >= 0.9, f"SmallCNN reached only {acc:.3f} on a pure location task"


def test_smallcnn_logits_shape() -> None:
    """Given a 64x48 batch, When forwarded, Then the logits are (N, 10)."""
    assert SmallCNN()(torch.zeros(2, 3, H, W)).shape == (2, NUM_CLASSES)
