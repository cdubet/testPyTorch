"""Fine-tune a ResNet-50 on DATASET/TRAIN + DATASET/VAL and classify images.

The training images are background-removed cutouts (RGBA, mostly transparent) that fill
the frame, while the real test photos (Test_Images) are large phone pictures of small
glassware on a desk, with real backgrounds, portrait orientation and EXIF rotation.
The data pipeline below is built to close that gap: cutouts are zoomed out and
composited onto randomised backgrounds, images are letterboxed instead of squashed,
EXIF orientation is honoured and augmentation is aggressive.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "DATASET" / "TRAIN"
VAL_DIR = BASE_DIR / "DATASET" / "VAL"
MODEL_PATH = BASE_DIR / "model.pth"
HISTORY_PATH = BASE_DIR / "training_history.json"
IMAGE_SIZE = 224

BATCH_SIZE = 16
EPOCHS = 30
FREEZE_EPOCHS = 2
BACKBONE_LR = 3e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
PATIENCE = 8
MIX_PROBABILITY = 0.5
ZOOM_OUT_PROBABILITY = 0.6
ARCHITECTURE = "resnet50"
LEGACY_ARCHITECTURE = "resnet18"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Letterbox colour: ImageNet mean, so padding is "neutral" after normalisation.
PAD_FILL = (124, 116, 104)
EVAL_BACKGROUND = (255, 255, 255)
IMAGE_EXTENSIONS = tuple(datasets.folder.IMG_EXTENSIONS)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False, dropout: float = 0.3) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.network = models.resnet50(weights=weights)
        self.network.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.network.fc.in_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def backbone_parameters(self):
        return [p for n, p in self.network.named_parameters() if not n.startswith("fc.")]

    def head_parameters(self):
        return list(self.network.fc.parameters())

    def load_checkpoint_state(self, state_dict: dict) -> None:
        # Checkpoints written before the Dropout head stored the linear layer as "fc.*".
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("network.fc.") and not key.startswith("network.fc.1."):
                key = key.replace("network.fc.", "network.fc.1.", 1)
            remapped[key] = value
        self.load_state_dict(remapped)


# --------------------------------------------------------------------------- images


def _random_background(size: tuple[int, int], rng) -> Image.Image:
    """Solid, gradient or noisy background so the model cannot rely on a black/transparent one."""
    width, height = size
    first = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    second = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
    kind = rng.random()
    if kind < 0.4:
        blend = np.zeros((height, width, 1), dtype=np.float32)
    elif kind < 0.7:
        blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None].repeat(width, 1)
    else:
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        blend = ((xs + ys) / 2.0)[:, :, None]
    pixels = first * (1.0 - blend) + second * blend
    noise_rng = np.random.default_rng(rng.getrandbits(32))
    pixels += noise_rng.normal(0.0, rng.uniform(0.0, 20.0), pixels.shape).astype(np.float32)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGB").convert("RGBA")


def _has_transparency(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )


def _pad_to_square(image: Image.Image, fill) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new(image.mode, (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _zoom_out(image: Image.Image, rng, min_scale: float = 0.3) -> Image.Image:
    """Shrink a square RGBA image and drop it somewhere on a transparent canvas of the same size.

    Test photos show small glassware on a desk; the cutouts fill the whole frame.
    The object centre stays in the middle half of the canvas so later random crops keep it.
    """
    side = image.width
    new_side = max(1, int(side * rng.uniform(min_scale, 1.0)))
    small = image.resize((new_side, new_side), Image.BILINEAR)
    low = max(0, side // 4 - new_side // 2)
    high = max(low, min(side - new_side, 3 * side // 4 - new_side // 2))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(small, (rng.randint(low, high), rng.randint(low, high)))
    return canvas


def load_image(path: Path, rng=None, zoom_out_probability: float = 0.0) -> Image.Image:
    """Open an image as RGB, honouring EXIF rotation.

    Inference (``rng`` is None): transparent pixels are composited onto white.
    Training/validation: the image is padded to a square, optionally zoomed out and
    composited onto a randomised background so a transparent/black background,
    the aspect ratio and the object scale cannot be used as shortcuts.
    """
    with Image.open(path) as image:
        if image.format == "JPEG":
            # Phone photos are huge; let the decoder downscale cheaply first.
            image.draft("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE * 3))
        image = ImageOps.exif_transpose(image)
        if rng is None:
            if _has_transparency(image):
                image = image.convert("RGBA")
                background = Image.new("RGBA", image.size, EVAL_BACKGROUND + (255,))
                image = Image.alpha_composite(background, image)
            return image.convert("RGB")
        image = _pad_to_square(image.convert("RGBA"), (0, 0, 0, 0))
        if rng.random() < zoom_out_probability:
            image = _zoom_out(image, rng)
        return Image.alpha_composite(_random_background(image.size, rng), image).convert("RGB")


class PadToSquare:
    """Letterbox to a square so tall phone photos are not squashed."""

    def __init__(self, fill: tuple[int, int, int] = PAD_FILL) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return _pad_to_square(image, self.fill)


def make_transforms(training: bool = False) -> transforms.Compose:
    if training:
        image_transforms = [
            PadToSquare(),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(20, fill=PAD_FILL)], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3, fill=PAD_FILL),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value="random"),
        ]
    else:
        image_transforms = [
            PadToSquare(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(image_transforms)


class ReadableImageFilter:
    """ImageFolder ``is_valid_file`` hook that drops unreadable/corrupt files and remembers them."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def __call__(self, path: str) -> bool:
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            return False
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            self.skipped.append(path)
            return False
        return True


class GlasswareDataset(datasets.ImageFolder):
    def __init__(self, root: Path, training: bool) -> None:
        self.filter = ReadableImageFilter()
        super().__init__(str(root), transform=make_transforms(training), is_valid_file=self.filter)
        self.training = training

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if self.training:
            rng = random
        else:
            # Stable per-file background/zoom so validation numbers are comparable across epochs.
            digest = hashlib.md5(path.encode("utf-8")).hexdigest()
            rng = random.Random(int(digest[:8], 16))
        image = load_image(Path(path), rng, ZOOM_OUT_PROBABILITY)
        return self.transform(image), target

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.classes)
        for _, target in self.samples:
            counts[target] += 1
        return counts


# -------------------------------------------------------------------------- training


def _set_backbone_trainable(model: ImageClassifier, trainable: bool) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(trainable)


def _evaluate(model, loader, loss_function, device, num_classes, use_amp):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_function(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1
    total = int(confusion.sum().item())
    accuracy = confusion.diag().sum().item() / max(total, 1)
    per_class_recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()
    balanced_accuracy = per_class_recall.mean().item()
    return total_loss / max(total, 1), accuracy, balanced_accuracy, confusion


def train_model(args: argparse.Namespace) -> None:
    if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected training and validation folders at {TRAIN_DIR} and {VAL_DIR}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = GlasswareDataset(TRAIN_DIR, training=True)
    val_dataset = GlasswareDataset(VAL_DIR, training=False)
    if not train_dataset.classes:
        raise ValueError(f"No class folders found in {TRAIN_DIR}.")
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Training and validation folders must contain the same class folders. "
            f"Training: {train_dataset.classes}; validation: {val_dataset.classes}"
        )
    skipped = train_dataset.filter.skipped + val_dataset.filter.skipped
    if skipped:
        print(f"Skipped {len(skipped)} unreadable image files:")
        for path in skipped:
            print(f"  {path}")

    classes = train_dataset.classes
    num_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    # Oversample rare classes so every class is seen equally often.
    class_counts = train_dataset.class_counts()
    sample_weights = [1.0 / class_counts[target] for _, target in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, drop_last=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = ImageClassifier(num_classes, pretrained=True).to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    mixer = v2.RandomChoice([v2.CutMix(num_classes=num_classes), v2.MixUp(num_classes=num_classes, alpha=0.4)])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.backbone_lr, "weight_decay": WEIGHT_DECAY},
            {"params": model.head_parameters(), "lr": args.head_lr, "weight_decay": 1e-4},
        ]
    )
    warmup_epochs = max(args.freeze_epochs, 1)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = []
    best_balanced_accuracy = -1.0
    best_epoch = 0
    best_confusion = None
    epochs_without_improvement = 0

    print(
        f"Training on {device}{' with AMP' if use_amp else ''}; "
        f"{len(train_dataset)} train / {len(val_dataset)} val images; classes: {', '.join(classes)}"
    )
    print(f"Train class counts: {dict(zip(classes, class_counts))}")

    for epoch in range(args.epochs):
        started = time.time()
        backbone_trainable = epoch >= args.freeze_epochs
        _set_backbone_trainable(model, backbone_trainable)
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            targets = labels
            if not args.no_mix and random.random() < MIX_PROBABILITY:
                images, targets = mixer(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = loss_function(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_seen += images.size(0)

        val_loss, val_accuracy, val_balanced_accuracy, confusion = _evaluate(
            model, val_loader, loss_function, device, num_classes, use_amp
        )
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        scheduler.step()

        train_loss = total_loss / max(train_seen, 1)
        train_accuracy = train_correct / max(train_seen, 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": val_balanced_accuracy,
                "backbone_trainable": backbone_trainable,
                "learning_rates": learning_rates,
                "seconds": time.time() - started,
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} ({time.time() - started:.0f}s) - "
            f"loss: {train_loss:.4f} - train accuracy: {train_accuracy:.1%} - "
            f"val loss: {val_loss:.4f} - val accuracy: {val_accuracy:.1%} - "
            f"val balanced accuracy: {val_balanced_accuracy:.1%}"
            f"{' (backbone frozen)' if not backbone_trainable else ''}"
        )

        if val_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = val_balanced_accuracy
            best_epoch = epoch + 1
            best_confusion = confusion.tolist()
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": ARCHITECTURE,
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "image_size": IMAGE_SIZE,
                    "preprocessing": "letterbox",
                    "epoch": best_epoch,
                    "validation_accuracy": val_accuracy,
                    "balanced_validation_accuracy": val_balanced_accuracy,
                },
                args.model,
            )
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(f"No improvement for {args.patience} epochs; stopping early.")
                break

    print(f"Best balanced validation accuracy: {best_balanced_accuracy:.1%} (epoch {best_epoch})")
    print(f"Model saved to {args.model}")

    HISTORY_PATH.write_text(
        json.dumps(
            {
                "classes": classes,
                "best_epoch": best_epoch,
                "best_balanced_validation_accuracy": best_balanced_accuracy,
                "confusion_matrix_best_epoch": best_confusion,
                "skipped_files": skipped,
                "epochs": history,
            },
            indent=2,
        )
    )
    _plot_history(history)


def _plot_history(history: list[dict]) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.figure()
    plt.plot(epochs, [e["train_loss"] for e in history], marker="o", label="Training")
    plt.plot(epochs, [e["val_loss"] for e in history], marker="o", label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, [e["train_accuracy"] for e in history], marker="o", label="Training")
    plt.plot(epochs, [e["val_accuracy"] for e in history], marker="o", label="Validation")
    plt.plot(
        epochs,
        [e["val_balanced_accuracy"] for e in history],
        marker="s",
        linestyle="--",
        label="Validation (balanced)",
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(BASE_DIR / "training_accuracy.png")
    plt.close()


# ------------------------------------------------------------------------- inference


def load_model(model_path: Path, device: torch.device) -> tuple[ImageClassifier, list[str]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}. Train one first with -train.")
    checkpoint = torch.load(model_path, map_location=device)
    # Older versions saved ResNet-50 weights under this incorrect label.
    if checkpoint.get("architecture") not in {ARCHITECTURE, LEGACY_ARCHITECTURE}:
        raise ValueError(
            "This model checkpoint was not created with ResNet-50. "
            "Train a new model with -train before classifying images."
        )
    classes = checkpoint["classes"]
    model = ImageClassifier(len(classes)).to(device)
    model.load_checkpoint_state(checkpoint["model_state"])
    model.eval()
    return model, classes


def predict_probabilities(
    model: ImageClassifier, image_path: Path, device: torch.device, tta: bool = True
) -> torch.Tensor:
    """Class probabilities for one image, averaged over the image and its mirror when ``tta``."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    tensor = make_transforms()(load_image(image_path))
    batch = torch.stack([tensor, tensor.flip(-1)]) if tta else tensor.unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(batch.to(device)), dim=1).mean(dim=0)


def classify_image(image_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(model_path, device)
    probabilities = predict_probabilities(model, image_path, device)
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
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="checkpoint path to write/read")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--freeze-epochs", type=int, default=FREEZE_EPOCHS,
        help="epochs to train only the classifier head before unfreezing the backbone",
    )
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    parser.add_argument(
        "--patience", type=int, default=PATIENCE,
        help="stop when balanced validation accuracy has not improved for this many epochs (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    parser.add_argument("--no-mix", action="store_true", help="disable CutMix/MixUp augmentation")
    args = parser.parse_args()

    if args.train:
        train_model(args)
    else:
        classify_image(args.image, args.model)


if __name__ == "__main__":
    main()