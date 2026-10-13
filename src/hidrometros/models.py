import torch.nn as nn
from torchvision import models

class SmallCNN(nn.Module):
    """CNN pequena de referência, treinada do zero para 10 classes."""
    def __init__(self, num_classes=10):
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

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

def build_model(name: str, pretrained: bool = True, num_classes: int = 10):
    """Cria o modelo e troca a camada final para 10 classes."""
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes)

    weights = "DEFAULT" if pretrained else None
    if name == "alexnet":
        model = models.alexnet(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "vgg16_bn":
        model = models.vgg16_bn(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Modelo desconhecido: {name}")
    return model

def freeze_backbone_for_first_epoch(model, name: str):
    """Congela backbone para os modelos transferidos; smallcnn não é congelada."""
    if name == "smallcnn":
        return
    for param in model.parameters():
        param.requires_grad = False

    if name in ("alexnet", "vgg16_bn"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif name == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
