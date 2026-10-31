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

# The batch size in a DataLoader determines how many samples are processed together in one iteration during training.
# Common practice suggests using a batch size between 8 and 128, depending on the dataset and model requirements.
BATCH_SIZE = 8
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6
ARCHITECTURE = "vgg19"


# ARCHITECTURE = "resnet18"
# https://github.com/MaitreyaM/Detailed-Image-Classification-Pytorch/blob/main/VGG19_classification.ipynb
# num_features=vgg19.classifier[6].in_features
# vgg19.classifier[6]=nn.Linear(num_features,6)
# vgg19=vgg19.to(device)
# criterion=nn.CrossEntropyLoss()
# optimizer=optim.Adam(vgg19.classifier[6].parameters(),lr=0.001)

class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        self.network = models.vgg19(weights=weights)
        self.network.classifier[6] = nn.Linear(
            self.network.classifier[6].in_features, num_classes
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def make_transforms(training: bool = False) -> transforms.Compose:
    image_transforms = []
    if training:
        image_transforms.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        )
    image_transforms.extend(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # TODO test with transforms.CenterCrop(224),
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

    num_classes = len(train_dataset.classes)
    train_class_counts = torch.bincount(
        torch.tensor(train_dataset.targets), minlength=num_classes
    )
    val_class_counts = torch.bincount(
        torch.tensor(val_dataset.targets), minlength=num_classes
    )
    if (train_class_counts == 0).any() or (val_class_counts == 0).any():
        raise ValueError(
            "Every class must contain images in both training and validation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageClassifier(num_classes, pretrained=True).to(device)
    class_weights = (
            len(train_dataset) / (num_classes * train_class_counts.float())
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    backbone_parameters = [
        parameter
        for name, parameter in model.network.named_parameters()
        if not name.startswith("classifier.6.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": model.network.classifier[6].parameters(), "lr": 1e-3},
        ]
    )
    # ReduceLROnPlateau is a learning rate scheduler used in machine learning to adjust
    # the learning rate based on the performance of a model.
    # It is particularly useful when training models that may experience plateaus
    # in their learning process.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # The learning rate is reduced when the monitored metric stops decreasing.
        factor=0.2,  # , the new learning rate will be 20% of the current rate.
        patience=2,  # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=1e-7  # The lower bound on the learning rate.
    )
    best_balanced_accuracy = -1.0
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0

    # store for printing
    losses = []
    accuracies = []
    validation_losses = []

    print(f"Training on {device}; classes: {', '.join(train_dataset.classes)}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_loss_weight = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            batch_weight = class_weights[labels].sum().item()
            total_loss += loss.item() * batch_weight
            total_loss_weight += batch_weight

        model.eval()
        correct = 0
        validation_class_correct = torch.zeros(num_classes, dtype=torch.long)
        validation_loss_total = 0.0
        validation_loss_weight = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels_on_device = labels.to(device)
                outputs = model(images.to(device))
                batch_weight = class_weights[labels_on_device].sum().item()
                validation_loss_total += loss_function(
                    outputs, labels_on_device
                ).item() * batch_weight
                validation_loss_weight += batch_weight
                predictions = outputs.argmax(dim=1).cpu()
                matches = predictions == labels
                correct += matches.sum().item()
                validation_class_correct += torch.bincount(
                    labels[matches], minlength=num_classes
                )

        validation_accuracy = correct / len(val_dataset)
        balanced_accuracy = (
                validation_class_correct.float() / val_class_counts
        ).mean().item()
        average_loss = total_loss / total_loss_weight
        average_validation_loss = validation_loss_total / validation_loss_weight
        losses.append(average_loss)
        accuracies.append(validation_accuracy)
        validation_losses.append(average_validation_loss)
        scheduler.step(average_validation_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {average_loss:.4f} - val loss: {average_validation_loss:.4f} - "
            f"val accuracy: {validation_accuracy:.1%} - "
            f"balanced accuracy: {balanced_accuracy:.1%}"
        )

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "classes": train_dataset.classes,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    print(f"Best validation balanced accuracy: {best_balanced_accuracy:.1%}")
    print(f"Accuracy at selected checkpoint: {best_validation_accuracy:.1%}")
    print(f"Model saved to {MODEL_PATH}")

    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, marker="o", label="Training")
    plt.plot(epochs, validation_losses, marker="o", label="Validation")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("training_loss_curve.png")
    plt.close()

    plt.plot(epochs, accuracies, marker="o")
    plt.title("accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
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
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(
            "This model checkpoint was not created with VGG-19. "
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
