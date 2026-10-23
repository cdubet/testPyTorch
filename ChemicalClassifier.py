import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ImageClassifier_Resnet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.network = models.resnet18(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    #TODO see impact of
    #def extract_features(self, x):
    #    feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
    #    with torch.no_grad():
    #        x = feature_extractor(x)
    #        x = x.view(x.size(0), -1)
    #    return x