import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: training runs without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "DATASET" / "TRAIN"
VAL_DIR = BASE_DIR / "DATASET" / "VAL"
MODEL_PATH = BASE_DIR / "model.pth"
HISTORY_PATH = BASE_DIR / "training_history.json"
IMAGE_SIZE = 224
ARCHITECTURE = "vgg19"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    """Hyper-parameters for train_model(); every field can be overridden from the CLI."""

    # Batch size: 8-128 is common; VGG-19 at 224px is memory hungry so the default stays small.
    batch_size: int = 8
    epochs: int = 30
    early_stopping_patience: int = 6
    # Epochs during which only the new classification head is trained. Lets the random head
    # settle before gradients flow into the pretrained backbone.
    freeze_backbone_epochs: int = 1
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # Set the maximum norm for gradient clipping, which is critical for preventing gradients from exploding during
    # backpropagation.Default is 5.0.
    max_grad_norm: float = 5.0

    num_workers: int = min(8, os.cpu_count() or 1)
    seed: int = 42
    use_amp: bool = True  # mixed precision, only takes effect on CUDA
    model_path: Path = MODEL_PATH


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    # Each DataLoader worker gets its own deterministic seed derived from the base seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def make_transforms(training: bool = False) -> transforms.Compose:
    if training:
        # Resize first so the geometric/colour augmentations run on small images,
        # then take a random crop of the slightly oversized image for translation jitter.
        padded_size = IMAGE_SIZE + IMAGE_SIZE // 8
        image_transforms = [
#            transforms.Resize((padded_size, padded_size)),
#            transforms.RandomCrop(IMAGE_SIZE),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    else:
        image_transforms = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(image_transforms)


def drop_unreadable_images(dataset: datasets.ImageFolder, name: str) -> None:
    """Remove samples PIL cannot decode so a corrupt file does not abort training mid-epoch."""
    readable = []
    dropped = torch.zeros(len(dataset.classes), dtype=torch.long)
    for path, target in dataset.samples:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            dropped[target] += 1
        else:
            readable.append((path, target))
    if dropped.sum() == 0:
        return
    dataset.samples = dataset.imgs = readable
    dataset.targets = [target for _, target in readable]
    report = ", ".join(
        f"{dataset.classes[i]}: {int(count)}" for i, count in enumerate(dropped) if count
    )
    print(f"Warning: skipping {int(dropped.sum())} unreadable {name} images ({report}).")


def load_datasets() -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
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
    drop_unreadable_images(train_dataset, "training")
    drop_unreadable_images(val_dataset, "validation")
    return train_dataset, val_dataset


def class_counts(dataset: datasets.ImageFolder) -> torch.Tensor:
    return torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes))


def make_data_loaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = dict(
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # no gradients: larger batches fit in memory
        shuffle=False,
        **loader_options,
    )
    return train_loader, val_loader


def run_training_epoch(
    model: ImageClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[float, float]:
    """Return the class-weighted mean loss and the plain accuracy over the epoch."""
    model.train()
    total_loss = 0.0
    total_loss_weight = 0.0
    correct = 0
    use_amp = scaler.is_enabled()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # CrossEntropyLoss(weight=...) returns a weighted mean; undo the division to sum over the epoch.
        batch_weight = class_weights[labels].sum().item()
        total_loss += loss.item() * batch_weight
        total_loss_weight += batch_weight
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / total_loss_weight, correct / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: ImageClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    class_weights: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """Return loss, accuracy, balanced accuracy, per-class accuracy and the confusion matrix."""
    model.eval()
    num_classes = len(loader.dataset.classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    total_loss_weight = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = loss_function(outputs, labels)
        batch_weight = class_weights[labels].sum().item()
        total_loss += loss.item() * batch_weight
        total_loss_weight += batch_weight
        predictions = outputs.argmax(dim=1)
        confusion += torch.bincount(
            labels * num_classes + predictions, minlength=num_classes**2
        ).reshape(num_classes, num_classes).cpu()

    per_class_accuracy = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    return {
        "loss": total_loss / total_loss_weight,
        "accuracy": confusion.diag().sum().item() / confusion.sum().item(),
        "balanced_accuracy": per_class_accuracy.mean().item(),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def print_class_report(classes: list[str], evaluation: dict) -> None:
    width = max(len(name) for name in classes)
    print(f"  {'class':<{width}}  accuracy  predicted as -> {' '.join(f'{c[:6]:>6}' for c in classes)}")
    for index, name in enumerate(classes):
        row = " ".join(f"{count:>6}" for count in evaluation["confusion_matrix"][index])
        print(f"  {name:<{width}}  {evaluation['per_class_accuracy'][index]:>7.1%}  {row}")


def save_plots(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], marker="o", label="Training")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Training")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation")
    plt.plot(epochs, history["val_balanced_accuracy"], marker="s", label="Validation (balanced)")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_accuracy.png")
    plt.close()


def train_model(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    set_seed(config.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick the fastest kernels

    train_dataset, val_dataset = load_datasets()
    classes = train_dataset.classes
    num_classes = len(classes)
    train_class_counts = class_counts(train_dataset)
    val_class_counts = class_counts(val_dataset)
    if (train_class_counts == 0).any() or (val_class_counts == 0).any():
        raise ValueError("Every class must contain images in both training and validation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.use_amp and device.type == "cuda"
    train_loader, val_loader = make_data_loaders(train_dataset, val_dataset, config, device)

    model = ImageClassifier(num_classes, pretrained=True).to(device)
    # Inverse-frequency weights so under-represented classes are not ignored.
    class_weights = (len(train_dataset) / (num_classes * train_class_counts.float())).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": config.backbone_lr},
            {"params": list(model.head_parameters()), "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    # Reduce the learning rate by 5x when validation loss stops improving for 2 epochs.
    # ReduceLROnPlateau is a learning rate scheduler used in machine learning to adjust
    # the learning rate based on the performance of a model.
    # It is particularly useful when training models that may experience plateaus
    # in their learning process.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = {
        key: []
        for key in (
            "train_loss", "train_accuracy", "val_loss", "val_accuracy",
            "val_balanced_accuracy", "backbone_lr", "head_lr", "epoch_seconds",
        )
    }
    best_epoch = 0
    best_evaluation: dict | None = None
    epochs_without_improvement = 0

    print(f"Training on {device} (mixed precision: {use_amp}); classes: {', '.join(classes)}")
    print(f"Train images per class: {train_class_counts.tolist()}; validation: {val_class_counts.tolist()}")
    print(f"Config: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()})}")

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        model.set_backbone_trainable(epoch >= config.freeze_backbone_epochs)
        if epoch == config.freeze_backbone_epochs and config.freeze_backbone_epochs > 0:
            print("Unfreezing the backbone.")

        train_loss, train_accuracy = run_training_epoch(
            model, train_loader, loss_function, class_weights, optimizer, scaler, device, config
        )
        evaluation = evaluate(model, val_loader, loss_function, class_weights, device, use_amp)
        scheduler.step(evaluation["loss"])
        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(evaluation["loss"])
        history["val_accuracy"].append(evaluation["accuracy"])
        history["val_balanced_accuracy"].append(evaluation["balanced_accuracy"])
        history["backbone_lr"].append(optimizer.param_groups[0]["lr"])
        history["head_lr"].append(optimizer.param_groups[1]["lr"])
        history["epoch_seconds"].append(epoch_seconds)

        print(
            f"Epoch {epoch + 1}/{config.epochs} ({epoch_seconds:.0f}s) - "
            f"loss: {train_loss:.4f} - acc: {train_accuracy:.1%} - "
            f"val loss: {evaluation['loss']:.4f} - val acc: {evaluation['accuracy']:.1%} - "
            f"val balanced acc: {evaluation['balanced_accuracy']:.1%} - "
            f"lr: {optimizer.param_groups[0]['lr']:.1e}/{optimizer.param_groups[1]['lr']:.1e}"
        )

        if best_evaluation is None or evaluation["balanced_accuracy"] > best_evaluation["balanced_accuracy"]:
            best_evaluation = evaluation
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "epoch": best_epoch,
                    "val_balanced_accuracy": evaluation["balanced_accuracy"],
                    "val_accuracy": evaluation["accuracy"],
                },
                config.model_path,
            )
            print(f"  New best model saved to {config.model_path}")
            print_class_report(classes, evaluation)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    history["best_epoch"] = best_epoch
    history["classes"] = classes
    history["best_evaluation"] = best_evaluation
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    save_plots(history)

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation balanced accuracy: {best_evaluation['balanced_accuracy']:.1%}")
    print(f"Accuracy at selected checkpoint: {best_evaluation['accuracy']:.1%}")
    print(f"Model saved to {config.model_path}; history saved to {HISTORY_PATH}")


def classify_image(image_path: Path, model_path: Path = MODEL_PATH) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train one first with -train."
        )
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
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
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="checkpoint path")

    defaults = TrainingConfig()
    training = parser.add_argument_group("training options")
    training.add_argument("--epochs", type=int, default=defaults.epochs)
    training.add_argument("--batch-size", type=int, default=defaults.batch_size)
    training.add_argument("--patience", type=int, default=defaults.early_stopping_patience,
                          help="early stopping patience in epochs")
    training.add_argument("--freeze-epochs", type=int, default=defaults.freeze_backbone_epochs,
                          help="epochs with a frozen backbone before fine-tuning everything")
    training.add_argument("--backbone-lr", type=float, default=defaults.backbone_lr)
    training.add_argument("--head-lr", type=float, default=defaults.head_lr)
    training.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    training.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    training.add_argument("--workers", type=int, default=defaults.num_workers,
                          help="DataLoader worker processes (0 = load in the main process)")
    training.add_argument("--seed", type=int, default=defaults.seed)
    training.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    args = parser.parse_args()

    if args.train:
        train_model(
            TrainingConfig(
                batch_size=args.batch_size,
                epochs=args.epochs,
                early_stopping_patience=args.patience,
                freeze_backbone_epochs=args.freeze_epochs,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
                num_workers=args.workers,
                seed=args.seed,
                use_amp=not args.no_amp,
                model_path=args.model,
            )
        )
    else:
        classify_image(args.image, args.model)


if __name__ == "__main__":
    main()
