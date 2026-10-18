import argparse
import time
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
#from sklearn.model_selection import train_test_split

import sys
import os

from torchvision import transforms, datasets

#from shoe_dataset import ShoeDataset
from ChemicalClassifier import ChemicalClassifier
from image_transforms import get_transforms_train, get_transforms_validation
from shared import get_num_classes

def train(epochs:int,save_model :bool,device:torch.device, model_name : str):
    since = time.time()

    data_transforms=dict()
    data_transforms['train'] = get_transforms_train()
    data_transforms['val']= get_transforms_validation()

    # based on https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    data_dir = '/home/christian/git/testPyTorch/DATASET'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    num_classes = get_num_classes()
    model = ChemicalClassifier(num_classes=num_classes)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    print("Start training...")

    #store for printing
    losses = []

    print("Number of batches:", len(dataloaders['train']))

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        for epoch in range(epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for images, labels in dataloaders[phase]:
                    images, labels = images.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item() * images.size(0)

                    # statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                if phase == 'train':
                    losses.append(epoch_loss)

                print(f"{phase} Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                     best_acc = epoch_acc
                     torch.save(model.state_dict(), best_model_params_path)

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')
            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

        plt.plot(range(1, epochs + 1), losses, marker="o")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        #plt.show()
        #TODO FigureCanvasAgg is non-interactive, and thus cannot be shown

        plt.savefig("training_loss_curve.png")

        torch.save(model.state_dict(), model_name)
        print("Saved model to "+model_name)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch chemical training')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--cpu', action='store_true',
                        help='for use cpu instead of cuda')
    parser.add_argument('-model_name', default='chemical-classifier.pth',
                        help='for use cpu instead of cuda')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.cpu):
        print("Running on CPU")
        device = torch.device("cpu")

    train(args.epochs,args.save_model,device, args.model_name)

if __name__ == '__main__':
        main()
