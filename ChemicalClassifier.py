import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ChemicalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ChemicalClassifier, self).__init__()
        # load the most up-to-date pre-trained weights
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        with torch.no_grad():
            x = feature_extractor(x)
            x = x.view(x.size(0), -1)
        return x