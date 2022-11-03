# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:34:58 2022

@author: antona

source: https://machinelearningmastery.com/lstm-autoencoders/

We can modify the reconstruction LSTM Autoencoder to instead predict the next step in the sequence.

Input sequence: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Prediction for each timestep: 0.1 -> 0.2 , 0.2 -> 0.3

[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

This means that the model will expect each input sequence to have nine time steps and the output sequence to have eight time steps.

For the last one 0.9 in the input sequence we do not know what to predict, that is why we get 8 predictions.
"""

# lstm autoencoder predict sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')

# fit model
model.fit(seq_in, seq_out, epochs=300, verbose=0)

# demonstrate prediction
print("Result:")
yhat = model.predict(seq_in, verbose=0)
print(yhat[0,:,0])