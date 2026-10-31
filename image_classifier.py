import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "DATASET" / "TRAIN"
VAL_DIR = BASE_DIR / "DATASET" / "VAL"
MODEL_PATH = BASE_DIR / "model.pth"
IMAGE_SIZE = 224

BATCH_SIZE = 8
EPOCHS = 30
ARCHITECTURE = "resnet50"
LEGACY_ARCHITECTURE = "resnet18"

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.network = models.resnet50(weights=weights)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

def make_transforms(training: bool = False) -> transforms.Compose:
    if training:
        image_transforms = [
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        ]
    else:
        image_transforms = [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(image_transforms)


def train_model() -> None:
    if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected training and validation folders at {TRAIN_DIR} and {VAL_DIR}."
        )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=make_transforms(True))
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=make_transforms())
    if not train_dataset.classes:
        raise ValueError(f"No class folders found in {TRAIN_DIR}.")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation folders must contain the same class folders. "
            f"Training: {train_dataset.classes}; validation: {val_dataset.classes}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = ImageClassifier(len(train_dataset.classes), pretrained=True).to(device)
    loss_function = nn.CrossEntropyLoss()
    backbone_parameters = [
        parameter
        for name, parameter in model.network.named_parameters()
        if not name.startswith("fc.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 5e-5},
            {"params": model.network.fc.parameters(), "lr": 5e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=3, min_lr=1e-7
    )
    best_accuracy = -1.0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print(f"Training on {device}; classes: {', '.join(train_dataset.classes)}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        model.eval()
        correct = 0
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                validation_loss += loss_function(outputs, labels).item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        validation_accuracy = correct / len(val_dataset)
        average_loss = total_loss / len(train_dataset)
        average_validation_loss = validation_loss / len(val_dataset)
        train_accuracy = train_correct / len(train_dataset)
        train_losses.append(average_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(average_validation_loss)
        val_accuracies.append(validation_accuracy)
        scheduler.step(validation_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {average_loss:.4f} - train accuracy: {train_accuracy:.1%} - "
            f"val loss: {average_validation_loss:.4f} - "
            f"val accuracy: {validation_accuracy:.1%}"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "classes": train_dataset.classes,
                    "image_size": IMAGE_SIZE,
                    "validation_accuracy": validation_accuracy,
                },
                MODEL_PATH,
            )

    print(f"Best validation accuracy: {best_accuracy:.1%}")
    print(f"Model saved to {MODEL_PATH}")

    epochs = range(1, EPOCHS + 1)
    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Training")
    plt.plot(epochs, val_losses, marker="o", label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("training_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, marker="o", label="Training")
    plt.plot(epochs, val_accuracies, marker="o", label="Validation")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("training_accuracy.png")
    plt.close()


# plt.show()
# TODO FigureCanvasAgg is non-interactive, and thus cannot be shown

def classify_image(image_path: Path) -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train one first with -train."
        )
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Older versions saved ResNet-50 weights under this incorrect label.
    if checkpoint.get("architecture") not in {ARCHITECTURE, LEGACY_ARCHITECTURE}:
        raise ValueError(
            "This model checkpoint was not created with ResNet-50. "
            "Train a new model with -train before classifying images."
        )
    classes = checkpoint["classes"]
    model = ImageClassifier(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_transform = make_transforms()
    with Image.open(image_path) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    print(
        f"Prediction: {classes[predicted_index]} "
        f"(confidence: {probabilities[predicted_index].item():.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an image classifier or classify one image."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-train", action="store_true", help="train and save a new model")
    mode.add_argument("-image", type=Path, help="image file to classify")
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        classify_image(args.image)


if __name__ == "__main__":
    main()