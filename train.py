import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split

import sys
import os

from torchvision import transforms, datasets

#from shoe_dataset import ShoeDataset
from ChemicalClassifier import ChemicalClassifier
from image_transforms import get_transforms_train, get_transforms_validation
from shared import get_num_classes

def train(epochs:int,save_model :bool,device:torch.device):
    #transform = get_transforms()

    #df = pd.read_csv("../data/Running Shoes Data.csv")
    #train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    #train_dataset = ShoeDataset(train_df, transform=transform)
    #train_data = [(img, label) for img, label in train_dataset if img is not None and label is not None]
    #train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
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

    print("Start training...")

    #store for printing
    losses = []

    print("Number of batches:", len(dataloaders['train']))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloaders['train'])
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
    plt.savefig("training_loss_curve.png")

    torch.save(model.state_dict(), "RunningShoesClassifier.pth")
    print("Saved model to RunningShoesClassifier.pth")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch chemical training')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    parser.add_argument('--cpu', action='store_true',
                        help='for use cpu instead of cuda')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.cpu):
        print("Running on CPU")
        device = torch.device("cpu")

    train(args.epochs,args.save_model,device)

if __name__ == '__main__':
        main()
