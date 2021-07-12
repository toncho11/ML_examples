# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
=============================
Classify images of clothing
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2

"""

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scale from [0..255] to [0..1]
train_images = train_images / 255.0
test_images = test_images / 255.0

#build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #no parameter learning just transforming from 28 x 28 to 784 pixels
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10) # Each node contains a score that indicates the current image belongs to one of the 10 classes
])

model.compile(optimizer='adam', #This is how the model is updated based on the data it sees and its loss function.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
              metrics=['accuracy']) #Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model.fit(train_images, train_labels, epochs=10)

#training results
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#The model's linear outputs, logits. Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
#Same model as the training one but with extra an extra layer Softmax to convert score to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#multiple predictions
predictions = probability_model.predict(test_images)

#single prediction
n = 1
img = test_images[n] #2 dimensions
img = (np.expand_dims(img,0)) #3 dimensions in order to create a 'batch'
predictions_single = probability_model.predict(img)
print("Classification of image ",n,": " , np.argmax(predictions_single[0]), class_names[np.argmax(predictions_single[0])])
if test_labels[n] == np.argmax(predictions_single[0]):
    print("True")
else:
    print ("False")

#check visually
imgplot = plt.imshow(img[0,:, :])#remove the first dimension because img.shape = (1, 28, 28)
plt.show()