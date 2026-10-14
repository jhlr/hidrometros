import torch.nn as nn
def cb(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
def get(name: str) -> nn.Module:
    if name == "flat":  # same 3 convs, 2 pools, no global pooling: keep the 16x12 map
        return nn.Sequential(*cb(3,32), nn.MaxPool2d(2), *cb(32,64), nn.MaxPool2d(2), *cb(64,128), nn.Flatten(), nn.Linear(128*16*12, 10))
    if name == "pool3flat":  # third pool, 8x6 map then flatten
        return nn.Sequential(*cb(3,32), nn.MaxPool2d(2), *cb(32,64), nn.MaxPool2d(2), *cb(64,128), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128*8*6, 10))
    if name == "pool3gap2x2":  # coarse spatial grid kept: adaptive pool to (4,3)
        return nn.Sequential(*cb(3,32), nn.MaxPool2d(2), *cb(32,64), nn.MaxPool2d(2), *cb(64,128), nn.AdaptiveAvgPool2d((4,3)), nn.Flatten(), nn.Linear(128*12, 10))
    if name == "gap_maxpool3":  # GAP but pooled earlier (3 pools): bigger receptive field
        return nn.Sequential(*cb(3,32), nn.MaxPool2d(2), *cb(32,64), nn.MaxPool2d(2), *cb(64,128), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128, 10))
    raise KeyError(name)

_get = get
def get(name: str) -> nn.Module:  # noqa-free wrapper: adds the scaffold head as "gap"
    if name == "gap":  # the original scaffold SmallCNN (global average pooling head)
        return nn.Sequential(*cb(3,32), nn.MaxPool2d(2), *cb(32,64), nn.MaxPool2d(2), *cb(64,128), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(128, 10))
    return _get(name)
