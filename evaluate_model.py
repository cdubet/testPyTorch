import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from ChemicalClassifier import ChemicalClassifier
from image_transforms import get_transforms, get_transforms_validation
from sklearn.preprocessing import LabelEncoder
from shared import get_num_classes
import numpy as np

from PIL import Image, ImageOps

import logging
low_threshold=0

logger = logging.getLogger(__name__)

option_tensorflow=false

def prepare_image(image_path: Path, width: int, height: int) -> np.ndarray:
    if option_tensorflow:
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)

        data = np.expand_dims(image_array, axis=0)
        return data
    norm_means = [0.485, 0.456, 0.406]
    norm_stdev = [0.229, 0.224, 0.225]
    pil_image = Image.open(image_path)

    #training was made with transforms.RandomResizedCrop(224),
    pil_image_resized = pil_image.resize((width, height))

    np_image = np.array(pil_image_resized)
    np_image2 = np_image.transpose(2, 0, 1)
    # print(np_image2[0])
    normalized_np_image = np.ndarray([3, width, height])
    for channel in range(len(np_image2)):
        normalized_np_image[channel] = ((np_image2[channel] / 255) - norm_means[channel]) / norm_stdev[channel]
        # normalized_np_image[channel] = np_image2[channel]
    return normalized_np_image

def predict(image_path, model, width: int, height: int):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    #TODO device = 'cuda'
    device='cpu'
    #model.to(device)
    # Create a dictionary of names indexed by model keys TODO make it in a external file
    model_idx_to_name=dict()
    model_idx_to_name[0]='becher'
    model_idx_to_name[1] = 'erlenmeyer'
    model_idx_to_name[2] = 'kolben'
    model_idx_to_name[3] = 'messzylinder'
    model_idx_to_name[4] = 'pipette'
    model_idx_to_name[5] = 'reagenzglas'

    #model_idx_to_name = {}
    #TODO store in model
    #for mod_key in model.class_to_idx.keys():
    #    model_idx_to_name[model.class_to_idx[mod_key]] = cat_to_name[mod_key]
    images = np.ndarray([1, 3, width, height])

    images[0] = prepare_image(image_path, width, height)
    image = torch.from_numpy(images).to(device)
    # image = [image.type(torch.cuda.FloatTensor) if device == 'cuda' else image.type(torch.FloatTensor)]
    #image = image.type(torch.cuda.FloatTensor)
    image = image.type(torch.FloatTensor)
    with torch.no_grad():
        output = model(image)
        top5 = output.topk(5)
        probs = top5[0].to('cpu').numpy()[0]
        classes = list(map(lambda x: model_idx_to_name[x], top5[1].to('cpu').numpy()[0]))

        return (probs, classes)

def normalize_root_dir_value(raw_root: str) -> str:
    return raw_root.strip().strip('"').strip("'")

def get_model(model_file_name):
    #check files exists
    if not Path(model_file_name).exists:
        print("model does not exist")
        return

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device('cpu')
    num_classes = get_num_classes() #38 # 6
    loaded_model = ChemicalClassifier(num_classes)  # TODO remove hard coded
    loaded_model.to(device)
    #toto=torch.load("/home/christian/git/RunningShoesClassifier/outputs/models/RunningShoesClassifier.pth")
    loaded_model.load_state_dict(torch.load(model_file_name, map_location=device))

    #set in evaluation mode
    loaded_model.eval()
    return loaded_model

def evaluate_model(config_path: str, 
                   model_path: str, 
                   labels_path: str, 
                   output_path: str  ) -> str:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")


    model = get_model(model_path)
    img_height = 256
    img_width = 256


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

        #data = prepare_image(image_path, img_width, img_height)
        prediction =  predict(image_path, model, img_width, img_height)

        got = "jdfjfdkh"
        confidence = 100

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
        default="chemical-classifier.pth",
        help="Path to pytorch model file.",
    )
    parser.add_argument(
        "-labels",
        default="labels.txt",
        help="Path to labels file.",
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
        labels_path=args.labels,
        output_path=args.output,
    )

    print(f"Test result written to: {output_file}")


if __name__ == "__main__":
    main()
