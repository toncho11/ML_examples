# -*- coding: utf-8 -*-
"""

Signal Processing with TensorFlow
Time series prediction model

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2

Tutorial https://towardsdatascience.com/machine-learning-and-signal-processing-103281d27c4b

"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
#import scipy.fftpack
import sys

import tensorflow as tf
from tensorflow import keras

print("Python version is: ",sys.version)
print("TensorFlow version is: ", tf.__version__,"\n")

def dnn_keras_tspred_model():
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  
  optimizer = tf.keras.optimizers.Adam()
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae']) 
  model.summary()
  
  return model

# Generate data
num_train_data = 4000
num_test_data = 1000
timestep = 0.1
tm =  np.arange(0, (num_train_data+num_test_data)*timestep, timestep);
y = np.sin(tm) + np.sin(tm*np.pi/2) + np.sin(tm*(-3*np.pi/2)) 
SNR = 10
ypn = y + np.random.normal(0,10**(-SNR/20),len(y))

plt.plot(tm[0:100],y[0:100])
plt.plot(tm[0:100],ypn[0:100],'r')
plt.show()
print("Red is the signal with noise.")

# prepare the train_data and train_labels
dnn_numinputs = 64
num_train_batch = 0
train_data = []
for k in range(num_train_data-dnn_numinputs-1):
  train_data = np.concatenate((train_data,ypn[k:k+dnn_numinputs]));
  num_train_batch = num_train_batch + 1
  
train_data = np.reshape(train_data, (num_train_batch,dnn_numinputs))
train_labels = y[dnn_numinputs:num_train_batch+dnn_numinputs]

#build model
model = dnn_keras_tspred_model()

#train model
EPOCHS = 100

model.fit(train_data, train_labels, epochs=EPOCHS,
          validation_split=0.2, verbose=0,
          callbacks=[])

# test how well DNN predicts now
num_test_batch = 0
strt_idx = num_train_batch
test_data=[]
for k in range(strt_idx, strt_idx+num_test_data-dnn_numinputs-1):
  test_data = np.concatenate((test_data,ypn[k:k+dnn_numinputs]));
  num_test_batch = num_test_batch + 1  
test_data = np.reshape(test_data, (num_test_batch, dnn_numinputs))
test_labels = y[strt_idx+dnn_numinputs:strt_idx+num_test_batch+dnn_numinputs]


#plot result, red is the real date
dnn_predictions = model.predict(test_data).flatten()
keras_dnn_err = test_labels - dnn_predictions
plt.plot(dnn_predictions[0:100],'g')
plt.plot(test_labels[0:100],'r')
plt.show()

print("Done. Red is the real signal. Green is the predicted.")