#example from https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/mnist/main.py
import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    #values are the same as in      transforms.Compose(...       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    norm_means = [0.485, 0.456, 0.406]
    norm_stdev = [0.229, 0.224, 0.225]
    pil_image = Image.open(image)

    #training was made with transforms.RandomResizedCrop(224),
    pil_image_resized = pil_image.resize((224, 224))

    np_image = np.array(pil_image_resized)
    np_image2 = np_image.transpose(2, 0, 1)
    # print(np_image2[0])
    normalized_np_image = np.ndarray([3, 224, 224])
    for channel in range(len(np_image2)):
        normalized_np_image[channel] = ((np_image2[channel] / 255) - norm_means[channel]) / norm_stdev[channel]
        # normalized_np_image[channel] = np_image2[channel]
    return normalized_np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    #TODO device = 'cuda'
    device='cpu'
    #model.to(device)
    # Create a dictionary of names indexed by model keys TODO make it in a external file
    cat_to_name=dict()
    cat_to_name[0]='becher'
    cat_to_name[1] = 'erlenmeyer'

    model_idx_to_name = {}
    for mod_key in model.class_to_idx.keys():
        model_idx_to_name[model.class_to_idx[mod_key]] = cat_to_name[mod_key]
    images = np.ndarray([1, 3, 224, 224])

    images[0] = process_image(image_path)
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

def predict_from_image(image_path, model, topk=5):
    transform = get_transforms_validation()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None
    device="cpu"
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        _, predicted = torch.max(output, 1)

    predicted_idx = predicted.item()
    #index
    print  (predicted_idx)
    print (top5_catid)
    print (probabilities)
    return predicted_idx
   # predicted_model = label_encoder.inverse_transform([predicted_idx])[0]

def process_image_with_model(model_file_name,image):
    #check files exists
    if not Path(image).exists():
        print("image does not exist")
        return
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

    #probs, classes =
    predict_from_image(image,loaded_model)
    #print(probs)
    #print(classes)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='use PyTorch model ')

    parser.add_argument("-image", nargs='?')
    parser.add_argument("-model", nargs='?')

    try:
        args = parser.parse_args()
        if args.image is None:
            print("missing image argument")
            parser.print_help()
        else:
            if args.model is None:
                print("missing model name")
                parser.print_help()
            print('processing '+ args.image+ ' with model '+ args.model)
            process_image_with_model(args.model,args.image)

    except (argparse.ArgumentError or argparse.ArgumentTypeError):
        print("failed to parse arguments")
        parser.print_help()
        raise


if __name__ == '__main__':
    main()
