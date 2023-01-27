# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:57:24 2023

source: https://realpython.com/generative-adversarial-networks/
   
Handwritten Digits Generator With a GAN using the MNIST dataset

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

PyTotch Training Loop:

 - Forward propagation — compute the predicted y and calculate the current loss e.g.: loss_discriminator = loss_function(...)
 - After each epoch we set the gradients to zero before starting to do backpropagation e.g.: .zero_grad()
 - Perfrom the back propagation: loss.backaward()
 - Gradient descent — Finally, we will update model parameters by calling optimizer.step() function

torch version: 1.11.0+cu113
torch vision version: 0.12.0+cu113
"""

import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)

device =  device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

#prepare images
#1) converts the data to a PyTorch tensor [0 1]
#but the background is black and so most of the coefficients are equal to 0 when they’re represented using this range.
#2) changes the range of the coefficients from [0 1] to -1 to 1 by subtracting 0.5 from the original coefficients and dividing the result by 0.5
#With this transformation, the number of elements equal to 0 in the input samples is dramatically reduced, which helps in training the models.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))] #two tuples where each tuple has size = number of channels in the image (here 1), https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
)

#first create a train_set object
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

batch_size = 32
#next build a data loader from the train set object
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

#visualization of the training data
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r") #inverts the colors
    plt.xticks([])
    plt.yticks([])
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024), #input images 28 x 28 = 784
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1), #1 neuron that tells us the probability of the image belonging to the real training data.
            nn.Sigmoid(), 
        )

    def forward(self, x):
        x = x.view(x.size(0), 784) #vectorization of the input image
        output = self.model(x)
        return output
    
discriminator = Discriminator().to(device=device)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), #we decide to set the input latent space to 100
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784), #output (last layer) is an "image" 28 x 28 = 784
            nn.Tanh(), #since the output coefficients should be in the interval from -1 to 1 
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output

generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50 #default 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        
        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(  #real samples are labeled with 1
            device=device
        )
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to( #generated are labeled with 0
            device=device
        )
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad() #in PyTorch, it’s necessary to clear the gradients at each training step to avoid accumulating them.
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        #You calculate the gradients and update the descriminator weights.
        loss_discriminator.backward() 
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 100)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        #You calculate the gradients and update the generator weights.
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            
#Checking the Samples Generated by the GAN
#The output should be digits resembling the training data
latent_space_samples = torch.randn(batch_size, 100).to(device=device)
generated_samples = generator(latent_space_samples)

generated_samples = generated_samples.cpu().detach() #you need to move the data back to the CPU and call .detach()

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

