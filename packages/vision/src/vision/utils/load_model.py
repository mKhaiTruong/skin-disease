import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet50

def _build_efficientnet(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    return model

def _build_resnet(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
    return model