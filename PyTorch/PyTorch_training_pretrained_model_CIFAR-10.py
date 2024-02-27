# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:10:25 2022

PyTorch classfication of the CIFAR-10 dataset using the pre-trained model vgg19.
The first layers of VGG19 are freezed and the last one is changed to output 10 classes instead of 1000 classes.

source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
This tutorial trains from zero.
More info on the pre-trining used here: https://stackoverflow.com/questions/65690251/how-to-use-vgg19-transfer-learning-pretraining

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
with 6000 images per class. There are 50000 training images and 10000 test images.

You need to restart Python kernel to run this code or clear all variables!!!
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print ("You need to restart Python kernel to run this code or clear all variables!")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    
    #load the pre-trained model and then modify the last layer
    model = torchvision.models.vgg19(pretrained=True)
    
    #freeze the all layers
    for param in model.parameters():
        param.requires_grad = False
   
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.classifier._modules['6'] = nn.Linear(4096, 10, bias=True)
    
    print(model.classifier) #show new network
    
    net = model
    #net.to(device) #switch to GPU
    
    #Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    
    #Train the network
    print('Started Training')
    for epoch in range(2):  # loop over the dataset multiple times
    
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data #CPU version
            #inputs, labels = data[0].to(device), data[1].to(device) #GPU version
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


    print('Finished Training')
    
    #Single prediction
    dataiter = iter(testloader)
    images, labels = dataiter.next()
   
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))
    print('True classes: ', ' '.join(f'{classes[labels[j]]:5s}'
                                  for j in range(4)))
    
    #Evaluate the entire dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')