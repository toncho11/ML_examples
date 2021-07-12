# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 09:49:13 2021

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0

"""
import sys
import tensorflow as tf

print("Python version is: ",sys.version)
print("TensorFlow version is: ", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#using the 'Sequential API' (defines a stack of layers), the alternative is the "Functional API'
#'Functional API' is for building DAGs
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

#predictions = model(x_train[:1]).numpy()
#print(predictions)
#print("===========================================")
#print(tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, acc = model.evaluate(x_test,  y_test, verbose=2)

print ("Accuracy on test is: ", acc)