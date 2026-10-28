import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sympy import false
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from ChemicalClassifier import ImageClassifier_Resnet18, ImageClassifier_vgg19
from image_transforms import  make_transforms
from sklearn.preprocessing import LabelEncoder
from shared import  get_image_width, get_image_height
import numpy as np


import seaborn as sns
from PIL import Image, ImageOps

import logging
low_threshold=0

logger = logging.getLogger(__name__)

def print_confusion_matrix (all_labels,all_preds,class_names):
    #based on https://github.com/MaitreyaM/Detailed-Image-Classification-Pytorch/blob/main/VGG19_classification.ipynb
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("confusion.png")

def get_model( model_path: Path,device):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train one first with -train."
        )

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]
    model = ImageClassifier_vgg19(len(classes)).to(device) #ImageClassifier_Resnet18(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model,classes

def classify_image(image_path: Path, model,device,classes) :
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_transform = make_transforms(get_image_height(),get_image_width(),false)
    with Image.open(image_path) as image:
        image_tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(device)

    #no_grad=no gradient
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    print(
        f"Prediction: {classes[predicted_index]} "
        f"(confidence: {probabilities[predicted_index].item():.1%})"
    )
    return classes[predicted_index],probabilities[predicted_index].item()

def normalize_root_dir_value(raw_root: str) -> str:
    return raw_root.strip().strip('"').strip("'")

def evaluate_model(config_path: str, 
                   model_path: str, 
                   output_path: str  ) -> str:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes= get_model(Path(model_path), device)

    tree = ET.parse(config_file)
    config_root = tree.getroot()

    root_dir_node = config_root.find("root_dir")
    root_dir_text = root_dir_node.text if root_dir_node is not None and root_dir_node.text else ""
    normalized_root_dir = normalize_root_dir_value(root_dir_text)

    result_root = ET.Element("test_result")
    result_root_dir = ET.SubElement(result_root, "root_dir")
    result_root_dir.text = root_dir_text

    success_node = ET.SubElement(result_root, "success")
    fail_node = ET.SubElement(result_root, "fail")

    total = 0
    passed = 0
    all_preds = []
    all_labels = []
    #list[str]

    for test_item in config_root.findall("test_item"):
        file_name = test_item.get("file", "")
        expected = test_item.get("expected", "")

        total += 1

        image_path = Path(normalized_root_dir) / file_name if normalized_root_dir else Path(file_name)
        if not image_path.exists():
            image_path = config_file.parent / file_name


        if not image_path.exists():
            ET.SubElement(
                fail_node,
                "test_item",
                {
                    "file": file_name,
                    "expected": expected,
                    "got": "FILE_NOT_FOUND",
                    "confidence_score": "0.00",
                },
            )
            continue


        print(f"Predicting for {image_path}...")
        got,confidence=classify_image(image_path, model, device, classes)

        all_preds.append(got.casefold())
        all_labels.append(expected.casefold())

        attrs = {
            "file": file_name,
            "expected": expected,
            "got": got,
            "confidence_score": f"{confidence:.2f}",
        }

        if got.casefold() == expected.casefold():
            ET.SubElement(success_node, "test_item", attrs)
            passed += 1
        else:
            ET.SubElement(fail_node, "test_item", attrs)

    accuracy = (passed / total * 100.0) if total else 0.0
    summary = ET.SubElement(result_root, "summary")
    summary.set("total", str(total))
    summary.set("passed", str(passed))
    summary.set("failed", str(total - passed))
    summary.set("accuracy", f"{accuracy:.2f}")

    if output_path is None:
        if config_file.suffix.lower() == ".xml":
            output_file = str(config_file.with_name(config_file.stem + "-result.xml"))
        else:
            output_file = str(config_file.with_name(config_file.name + "-result.xml"))
    else:
        output_file = output_path

    ET.indent(result_root, space="    ")
    result_tree = ET.ElementTree(result_root)
    result_tree.write(output_file, encoding="utf-8", xml_declaration=True)

    #TODO get the names from model
    class_names = classes
    print_confusion_matrix(all_labels, all_preds, class_names)

    return output_file


def main() -> None:
    #for ex
    # -model mchemical.pth  -labels tensorflow_labels.txt -config config-accuracy-test.xml

    parser = argparse.ArgumentParser(
        description="Evaluate a pytorch model with an XML config and produce a test_result XML file."
    )
    parser.add_argument(
        "-config",
        default="config-accuracy-test.xml",
        help="Path to input configuration XML file.",
    )
    parser.add_argument(
        "-model",
        default="model.pth",
        help="Path to pytorch model file.",
    )

    parser.add_argument(
        "-output",
        default=None,
        help="Path to output XML result. Default: <config-name>-result.xml",
    )

    args = parser.parse_args()

    output_file = evaluate_model(
        config_path=args.config,
        model_path=args.model,
        output_path=args.output,
    )

    print(f"Test result written to: {output_file}")


if __name__ == "__main__":
    main()
