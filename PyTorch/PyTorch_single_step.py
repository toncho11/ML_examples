# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:11:45 2022

In this example, we load a pretrained resnet18 model from torchvision. We create 
a random data tensor to represent a single image with 3 channels, and height & width
of 64, and its corresponding label initialized to some random values. Label in 
pretrained models has shape (1,1000).

"""

import torch
from torchvision.models import resnet18, ResNet18_Weights

#create data
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum()

#Backward propagation is kicked off when we call .backward() on the error tensor. 
#Autograd then calculates and stores the gradients for each model parameter in the parameter’s .grad attribute.
loss.backward() # backward pass

#We give the optimizer a model (its parameters) to work with
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#Finally, we call .step() to initiate gradient descent. The optimizer adjusts each
# parameter by its gradient stored in .grad.
optim.step() #gradient descent