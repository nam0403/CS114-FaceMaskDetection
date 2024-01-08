import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import shutil
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


import random


def get_all_files(directory): #function to get all files from directory
    paths = []
    for root, _, files in os.walk(directory):
              for f_name in files:
                    path = os.path.join(root, f_name) #get a file and add the total path
                    paths.append(path)
    return paths #Return the file paths
directory = 'dataset/archive'
paths = get_all_files(directory)
random.shuffle(paths)
# copy 20% of the data to test folder no_mask with shupple function
def copy_files(paths, source, destination, percentage):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for path in tqdm(paths[:int(len(paths) * percentage)]):

        shutil.copy(path, destination)
    print('copy completed')

# read image in folder train and creat a datagen for image augmentation
def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    return image

# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ImageFolder(root='dataset/train/', transform=train_transforms)
val_dataset = ImageFolder(root='dataset/validate/', transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


# create a model
model = models.mobilenet_v2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier part of the model
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.BatchNorm1d(num_ftrs),
    nn.Flatten(),
    nn.Linear(num_ftrs, 1000),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1000, 3),
    nn.Softmax(dim=1)
)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-4/30)

#### test zone ####
# inputs, labels = next(iter(train_loader))

# inputs = torch.tensor(inputs).float().to(device)
# inputs = inputs.permute(0,3,1,2)
# labels = torch.tensor(labels).float().to(device)


# running_loss = 0.0
# correct = 0
# total = 0
# optimizer.zero_grad()  # Zero the parameter gradients
# outputs = model(inputs)  # Forward pass
# loss = criterion(outputs, labels)  # Compute loss
# loss.backward()  # Backward pass
# optimizer.step()  # Optimize
# running_loss += loss.item()
# _, predicted = outputs.max(1)
# total += labels.size(0)
# labels = labels.argmax(1)
# print(labels)
# print(predicted)

# correct += (predicted == labels).sum().item()
# print(correct)
def train():
# Training
    EPOCHS = 30
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, EPOCHS))
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = torch.tensor(inputs).to(device)
            labels = torch.tensor(labels).to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'  Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}')
        # Validation (You can add a validation loop similarly)
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        print('Validation')
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = torch.tensor(val_inputs).to(device), torch.tensor(val_labels).to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        # Calculate average training and validation loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%')

    torch.save(model.state_dict(), 'model.pth')
    print('Finished Training')


if __name__ == '__main__':
    train()