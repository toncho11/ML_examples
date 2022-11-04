# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:22:31 2022

@author: antona

source: https://machinelearningmastery.com/lstm-autoencoders/

Creating a new encoder model from a trained auto-encoder.

Regardless of the method chosen (reconstruction, prediction, or composite), once the autoencoder has been fit, the decoder can be removed and the encoder can be kept as a standalone model.

The encoder can then be used to transform input sequences to a fixed length encoded vector.

We can do this by creating a new model that has the same inputs as our original model, and outputs directly from the end of encoder model, before the RepeatVector layer.

Running the example creates a standalone encoder model that could be used or saved for later use.

We demonstrate the encoder by predicting the sequence and getting back the 100 element output of the encoder.

"""

# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.models import Model
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

# define auto-encoder model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1))) #model.layers[0] <= we reuse this one in new_model
model.add(RepeatVector(n_in)) #model.layers[1]
model.add(LSTM(100, activation='relu', return_sequences=True)) #model.layers[2]
model.add(TimeDistributed(Dense(1))) #model.layers[3]

# compile model
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)

# create a new model that reuses the trained encoder part of the old autonecoder model
new_model = Model(inputs=model.inputs, outputs=model.layers[0].output)

plot_model(new_model, show_shapes=True, to_file='lstm_encoder.png')

# get the feature vector for the input sequence
yhat = new_model.predict(sequence)
print(yhat.shape)
print(yhat)

