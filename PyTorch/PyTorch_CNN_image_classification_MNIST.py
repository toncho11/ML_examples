# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:08:10 2022

@author: antona

Source: https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/

This example demonstrates a multi class classification of the MNIST dataset.
The MNIST database contains 60,000 training images and 10,000 testing images.
Images are 28 x 28 pixels grayscale (with only one channel)

"""

# pytorch cnn for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# model definition
class CNN(Module):
    
    # define model elements
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        
        '''
        The model architecture is: CONV2D, POOL2D, CONV2D, POOL2D, LINEAR, LINEAR
        A layer is a transformation. LINEAR and Conv2d are functions that apply linear transformation and 2D convolution (over an input signal composed of several input planes)
        ReLU after each CONV2D, no act() function after each POOL2D
        ReLU and Softmax are used for the last two LINEAR layers.
        '''
        
        # input to first hidden layer CONV2D
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        
        # first pooling layer POOL2D
        self.pool1 = MaxPool2d((2,2), stride=(2,2))
        
        # second hidden layer CONV2D
        self.hidden2 = Conv2d(32, 32, (3,3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        
        # second pooling layer POOL2D
        self.pool2 = MaxPool2d((2,2), stride=(2,2))
        
        # third hideen layer - fully connected layer LINEAR
        self.hidden3 = Linear(5*5*32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        
        # output layer LINEAR
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        
        #up to hear CONV2D, POOL2D, CONV2D, POOL2D are processed
        
        # flatten - converting the data into a 1-dimensional array for inputting it to the next layer
        X = X.view(-1, 4*4*50)
        
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        
        return X

# prepare the dataset
def prepare_data(path):
    
    '''
    Define standardization
    Note that the images are arrays of grayscale pixel data, therefore, we must add 
    a channel dimension to the data before we can use the images as input to the model.
    It is a good idea to scale the pixel values from the default range of 0-255 to 
    have a zero mean and a standard deviation of 1.
    Compose() - composes several 'transforms' together.
    ToTensor() - convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Normalize() - normalizes a tensor image with mean and standard deviation.
    '''
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) 
    
    # load dataset
    train = MNIST(path, train=True,  download=True, transform=trans)
    test  = MNIST(path, train=False, download=True, transform=trans)
    
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=64,   shuffle=True)
    test_dl  = DataLoader(test,  batch_size=1024, shuffle=False)
    
    return train_dl, test_dl

# train the model
def train_model(train_dl, model):
    
    # define the optimization
    criterion = CrossEntropyLoss() #used when we have more than 2 classes
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # enumerate epochs
    for epoch in range(10):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            
            # clear the gradients
            optimizer.zero_grad()
            
            # compute the model output
            yhat = model(inputs)
            
            # calculate loss
            loss = criterion(yhat, targets)
            
            # backward() method is used to compute the gradient during the backward pass in a neural network. The gradients are computed when this method is executed.
            loss.backward()
            
            # update model weights using the newly computed gradients from backward()
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    
    predictions, actuals = list(), list()
    
    for i, (inputs, targets) in enumerate(test_dl):
        
        # evaluate the model on the test set
        yhat = model(inputs)
        
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        
        # store
        predictions.append(yhat)
        actuals.append(actual)
    
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc

# prepare the data
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

# define the network
model = CNN(1) #the images are grayscaled, so only 1 channel, an image is usally represented as W x H x C (Width x Height x Channel). Color images normally have 3 channels.

# train the model
train_model(train_dl, model)

# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)