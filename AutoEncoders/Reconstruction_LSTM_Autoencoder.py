# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:32:17 2022

@author: antona

LSTM Autoencoders

source: https://machinelearningmastery.com/lstm-autoencoders/

The simplest LSTM autoencoder is one that learns to reconstruct each input sequence.

In this example we use a dataset of one sample of nine time steps and one feature (sequence variable)

"""

# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# define input sequence
sequence = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)

# save a image that explains the model
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

# demonstrate recreation
print("Result:")
yhat = model.predict(sequence, verbose=0)
print(yhat[0,:,0])