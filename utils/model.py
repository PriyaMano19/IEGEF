import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes=4, pretrained=True):
    """
    Build a pretrained ResNet50 model for chest X-ray classification.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet pretrained weights.

    Returns:
        model (torch.nn.Module)
    """

    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model