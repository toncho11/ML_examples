# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:53:20 2022

@author: antona

source 1: https://blog.keras.io/building-autoencoders-in-keras.html
source 2: https://github.com/lyeoni/keras-mnist-VAE/blob/master/keras-mnist-VAE.ipynb

The is an example of Variational autoencoder (VAE). We construct the VAR model using the Keras API. 
Then we train the model and use it  
    
"""

import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

# Load the MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#Here's our encoder network, mapping inputs to our latent distribution parameters:

original_dim = 28 * 28
intermediate_dim = 64
latent_dim = 2

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)

#An encoder network turns the input samples x into two parameters in a latent space, which we will note z_mean and z_log_sigma.
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

#We can use these parameters (z_mean and z_log_sigma) to sample new similar points from the latent space:

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

#Finally, we can map these sampled latent points back to reconstructed inputs:

# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model (encoder/decoder)
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

#We train the model using the end-to-end model, with a custom loss function: the sum of a reconstruction term, and the KL divergence regularization term.
reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

#We train our VAE on MNIST digits:  
epochs = 20 #default 100
vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_test, x_test))

#We look at the neighborhoods of different classes on the latent 2D plane:
#Each of these colored clusters is a type of digit. Close clusters are digits that are structurally similar (i.e. digits that share information in the latent space).

#x_test_encoded = encoder.predict(x_test, batch_size=32)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()

# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# We will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        
        #z point from the latent normal distribution that is assumed to generate the data
        z_sample = np.array([[xi, yi]])
        
        #generate sample from the z point
        x_decoded = decoder.predict(z_sample) #generate the sample in (1, 784)
        
        digit = x_decoded[0].reshape(digit_size, digit_size) #convert to 28 x 28
        
        #place on the figure
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()