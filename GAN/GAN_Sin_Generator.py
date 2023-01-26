# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:32:03 2023

Source: https://realpython.com/generative-adversarial-networks/
    
Generating sin samples

D - descriminator
G - generator

The GAN training process consists of a two-player minimax game in which D is adapted to 
minimize the discrimination error between real and generated samples, and G is adapted
to maximize the probability of D making a mistake.

Although the dataset containing the real data isn’t labeled, the training processes for D
and G are performed in a supervised way. At each step in the training, D and G have their 
parameters updated.

To train D, at each iteration you label some real samples taken from the training data as
1 and some generated samples provided by G as 0.

For each batch of training data containing labeled real and generated samples, you update
the parameters of D to minimize a loss function. After the parameters of D are updated, 
you train G to produce better generated samples. The output of G is connected to D, whose 
parameters are kept frozen 

"""

import torch
from torch import nn

import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda")

#1 Preparing the Training Data
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2)) #a tensor with dimensions of 1024 rows and 2 columns
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
] #the train set is a list with two tensors per item. First tensor contains x(random number) and sin(x). Second one is 0 and not used, required by PyTorch data loader.

batch_size = 32
#creating a pytorch data loader (and sampler)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

#plotting train data
#plt.plot(train_data[:, 0], train_data[:, 1], ".")

#Implementing the Discriminator
class Discriminator(nn.Module): #In PyTorch, the neural network models are represented by classes that inherit from nn.Module
    def __init__(self):
        super().__init__()
        #The discriminator is a model with a two-dimensional input and a one-dimensional output.
        self.model = nn.Sequential(
            nn.Linear(2, 256), #The input is two-dimensional, and the first hidden layer is composed of 256 neurons with ReLU activation.
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1), #1 output
            nn.Sigmoid(), #
        )

    def forward(self, x): #you use .forward() to describe how the output of the model is calculated.
        output = self.model(x)
        return output

discriminator = Discriminator().to(device=device)

#Implementing the Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16), #A model with a two-dimensional input, which will receive random points (z1, z2)
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2), #(in features, out features),a two-dimensional output that must provide (x1, xx2) points resembling those from the training data.
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

#Training the Models

lr = 0.001
num_epochs = 300 #how many repetitions of training using the whole training set will be performed
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
#while training the generator below, we keep the discriminator weights frozen since we create the optimizer_generator with its first argument equal to generator.parameters()
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#you need to implement a training loop in which training samples are fed to the models, and their weights are updated
#to minimize the loss function:
for epoch in range(num_epochs):
    
    for n, (real_samples, _) in enumerate(train_loader):
        
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1)) #real are labeled with 1
        latent_space_samples = torch.randn((batch_size, 2))
        
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)) #generated are labeled with 0
        
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad() #in PyTorch, it’s necessary to clear the gradients at each training step to avoid accumulating them.
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        #You calculate the gradients and update the descriminator weights.
        loss_discriminator.backward()  #backward propagation - updating the gradients
        optimizer_discriminator.step() #gradient descent

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator
        generator.zero_grad() #in PyTorch, it’s necessary to clear the gradients at each training step to avoid accumulating them.
        
        #You feed the generator’s output into the discriminator and store its output in 
        #output_discriminator_generated, which you’ll use as the output of the whole model.
        generated_samples = generator(latent_space_samples) #this data is different from when we trained the descriminator
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels #real_samples_labels are all 1s
        )
        
        #You calculate the gradients and update the generator weights.
        loss_generator.backward()  #backward propagation - updating the gradients
        optimizer_generator.step() #gradient descent

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            
#Checking the Samples Generated by the GAN
#The objective was to train the generator, so that we can transform latent space variables (x1,x2) into x and sin(x)
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".") #first column is x, second is sin(x)