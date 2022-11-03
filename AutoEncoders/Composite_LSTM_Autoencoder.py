# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:53:52 2022

@author: antona

source: https://machinelearningmastery.com/lstm-autoencoders/

We put two decoders after the encoder. The first oner is for reconstruction with 9 outputs.
Tge second one is for prediction with 8 values because we can not predict the last one.

"""

# lstm autoencoder reconstruct and predict sequence
from numpy import array
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import numpy as np

# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1

# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)

# define reconstruct decoder with 9 values
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)

# define predict decoder with 8 values
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)

# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')

plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')

# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=1)

# demonstrate prediction
print("Result:")
yhat = model.predict(seq_in, verbose=0)
print(yhat)
print(len(yhat[0][0])) #the reconstructud sequence
print(len(yhat[1][0])) #the prediction