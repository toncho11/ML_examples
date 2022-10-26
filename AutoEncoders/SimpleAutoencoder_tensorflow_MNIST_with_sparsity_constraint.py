# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:59:56 2022

@author: antona

source1: https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/
source2: https://blog.keras.io/building-autoencoders-in-keras.html

Learning the representation of the MNIST dataset. 

Here what typically happens is that the hidden layer is learning an approximation of PCA (principal component analysis).

to constrain the representations to be compact is to add a sparsity contraint on the activity of the 
hidden representations, so fewer units would "fire" at a given time. In Keras, this can be done by 
adding an activity_regularizer to our Dense layer.

With the added regularization the model is less likely to overfit and can be trained longer
"""

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

encoding_dim = 15 #the dimension of the encoding vector produced by the encoder,  compression of factor 32, assuming the input is 784 floats
#higher value of encoding_dim means better representation, less compression

input_img = Input(shape=(784,))
# Add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

# Model which take input image and shows decoded images
autoencoder = Model(input_img, decoded)

# This model shows encoded images
encoder = Model(input_img, encoded)
# Creating a decoder model
encoded_input = Input(shape=(encoding_dim,))
# last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Train the data
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                validation_data=(x_test, x_test))

# Execute on the test data set
encoded_img = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_img)

# Show result - first line are original images, second line are decoded images
plt.figure(figsize=(20, 4))
for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
