"""The four models and the shared training protocol table.

`SmallCNN`, `build_model` and the freeze helpers started as the group scaffold by Mikael de
Albuquerque Mulatinho and were corrected here (explicit IMAGENET1K_V1 weights, per-model LR groups,
typing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet18_Weights, VGG16_BN_Weights

MODEL_NAMES: tuple[str, ...] = ("smallcnn", "alexnet", "vgg16_bn", "resnet18")
NUM_CLASSES = 10


@dataclass(frozen=True)
class ModelSpec:
    """One row of the protocol table (assignment, Part 4)."""

    name: str
    pretrained: bool
    input_hw: tuple[int, int]  # (height, width)
    batch_cuda: int
    batch_mps_cpu: int
    lr_head: float
    lr_backbone: float | None
    max_epochs: int


PROTOCOL: dict[str, ModelSpec] = {
    "smallcnn": ModelSpec("smallcnn", False, (64, 48), 256, 128, 1e-3, None, 10),
    "alexnet": ModelSpec("alexnet", True, (128, 128), 128, 64, 1e-3, 1e-4, 8),
    "vgg16_bn": ModelSpec("vgg16_bn", True, (128, 128), 32, 16, 1e-3, 1e-5, 8),
    "resnet18": ModelSpec("resnet18", True, (128, 128), 128, 64, 1e-3, 1e-4, 8),
}

EARLY_STOP_PATIENCE = 3
LABEL_SMOOTHING = 0.1
SEED = 42


class SmallCNN(nn.Module):
    """Small reference CNN trained from scratch for 10 classes."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.classifier(self.features(x).flatten(1)))


class PortableAdaptiveAvgPool2d(nn.Module):
    """`nn.AdaptiveAvgPool2d` that also works on MPS when the output size does not divide the input.

    VGG16 at 128x128 reaches its pooling layer with a 4x4 map and needs 7x7 out; the MPS backend
    does not implement this non-divisible resize. On MPS and only then, the (tiny) tensor goes through the
    CPU, so numerics match the reference layer exactly.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if x.device.type == "mps" and (h, w) != (self.output_size, self.output_size):
            return torch.as_tensor(F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device))
        return torch.as_tensor(F.adaptive_avg_pool2d(x, self.output_size))


def build_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Create a model and replace its last layer by a `num_classes` head.

    Pretrained weights are the explicit torchvision `IMAGENET1K_V1` set (not "DEFAULT", which can
    point to a different recipe in newer torchvision versions).
    """
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if name == "alexnet":
        alex = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        alex.classifier[6] = nn.Linear(alex.classifier[6].in_features, num_classes)
        return cast(nn.Module, alex)
    if name == "vgg16_bn":
        vgg = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)
        vgg.avgpool = PortableAdaptiveAvgPool2d(7)
        return cast(nn.Module, vgg)
    if name == "resnet18":
        res = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        res.fc = nn.Linear(res.fc.in_features, num_classes)
        return cast(nn.Module, res)
    raise ValueError(f"unknown model: {name}")


def head_parameters(model: nn.Module, name: str) -> list[nn.Parameter]:
    """Parameters of the final classifier block (trained during the frozen first epoch)."""
    if name in ("alexnet", "vgg16_bn"):
        return list(model.get_submodule("classifier").parameters())
    if name == "resnet18":
        return list(model.get_submodule("fc").parameters())
    return list(model.parameters())


def backbone_parameters(model: nn.Module, name: str) -> list[nn.Parameter]:
    """Everything that is not the head (empty for smallcnn, which has a single LR group)."""
    if name == "smallcnn":
        return []
    head_ids = {id(p) for p in head_parameters(model, name)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_frozen(model: nn.Module, name: str, frozen: bool) -> None:
    """Freeze/unfreeze the backbone of pretrained models. smallcnn is never frozen.

    When frozen, BatchNorm layers of the backbone are also kept in eval mode by `train_mode`.
    """
    if name == "smallcnn":
        return
    for p in backbone_parameters(model, name):
        p.requires_grad = not frozen


def train_mode(model: nn.Module, name: str, frozen: bool) -> None:
    """model.train(), except frozen-backbone BatchNorm stays in eval (statistics not updated)."""
    model.train()
    if frozen and name != "smallcnn":
        head_ids = {id(p) for p in head_parameters(model, name)}
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                id(p) in head_ids for p in module.parameters()
            ):
                module.eval()


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
