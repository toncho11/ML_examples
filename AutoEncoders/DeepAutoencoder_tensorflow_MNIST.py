# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:59:56 2022

@author: antona

source1: https://www.analyticsvidhya.com/blog/2021/06/complete-guide-on-how-to-use-autoencoders-in-python/
source2: https://blog.keras.io/building-autoencoders-in-keras.html

Learning the representation of the MNIST dataset.

The encoder is made up of a stack of Conv2D and max-pooling layer and the decoder is a stack
of Conv2D and Upsampling Layer.

CNN produces better results.

"""

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model, Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

model = Sequential()
# encoder network
model.add(Conv2D(30, 3, activation= 'relu', padding='same', input_shape = (28,28,1)))
model.add(MaxPooling2D(2, padding= 'same'))
model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
model.add(MaxPooling2D(2, padding= 'same'))
# decoder network
model.add(Conv2D(15, 3, activation= 'relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(30, 3, activation= 'relu', padding='same'))
model.add(UpSampling2D(2))
model.add(Conv2D(1,3,activation='sigmoid', padding= 'same')) # output layer

# compile 
model.compile(optimizer= 'adam', loss = 'binary_crossentropy')

# Load the data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Train the data
model.fit(x_train, x_train,
                epochs=15,
                batch_size=128,
                validation_data=(x_test, x_test))


# Execute on the test data set
pred = model.predict(x_test)

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
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()
