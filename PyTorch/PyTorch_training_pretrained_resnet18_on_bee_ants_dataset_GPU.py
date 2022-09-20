# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:34:37 2022

source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

The problem we’re going to solve today is to train a model to classify ants and bees. We have about 120 training 
images each for ants and bees. There are 75 validation images for each class. Usually, this is a very small dataset 
to generalize upon, if trained from scratch. Since we are using transfer learning, we should be able to generalize reasonably well.

We are training a resnet18 pretrained model on a small bee-ants dataset.
Dataset is available either in ./data or https://download.pytorch.org/tutorial/hymenoptera_data.zip

The code here does not use the train function from the original example.

We just change the last layer to 2 classes, all the other layers are not freezed and will be adapted to the new dataset.

"""

from __future__ import print_function, division

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
import copy

print ("You need to restart Python kernel to run this code or clear all variables!")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)

trainloader = dataloaders['train']
testloader = dataloaders['val']

#Finetuning the convnet
#We do not freeze any layers, we only replace the last one: model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    #Train the network
    print('Started Training')
    
    for epoch in range(2):  # loop over the dataset multiple times - default 2
    
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data #CPU version
            inputs, labels = data[0].to(device), data[1].to(device) #GPU version
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model_ft(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    print('Finished Training')
    
    print("Started evaluating the entire test dataset:")
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    
    with torch.no_grad():
        for data in testloader:
            
            #images, labels = data #CPU version
            images, labels = data[0].to(device), data[1].to(device) #GPU version
            
            # calculate outputs by running images through the network
            outputs = model_ft(images)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')



