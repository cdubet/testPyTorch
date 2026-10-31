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
class ImageClassifier_vgg19(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        self.network = models.vgg19(weights=weights)
        self.network.classifier[6] = nn.Linear(
            self.network.classifier[6].in_features, num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def backbone_parameters(self):
        return (
            parameter
            for name, parameter in self.network.named_parameters()
            if not name.startswith("classifier.6.")
        )

    def head_parameters(self):
        return self.network.classifier[6].parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_parameters():
            parameter.requires_grad_(trainable)

class ImageClassifier_raw(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))