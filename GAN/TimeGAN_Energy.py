# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:44:45 2022

@author: antona

pip install ydata-synthetic

source: https://towardsdatascience.com/modeling-and-generating-time-series-data-using-timegan-29c00804f54d
source: https://github.com/archity/synthetic-data-gan/blob/main/timeseries-data/energy-data-synthesize.ipynb

"""

import os
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters

#Define Model Hyperparameters
# Specific to TimeGANs

seq_len = 24        # Timesteps
n_seq = 28          # Features

# Hidden units for generator (GRU & LSTM).
# Also decides output_units for generator
hidden_dim = 24

gamma = 1           # Used for discriminator loss

noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128

learning_rate = 5e-4
beta_1 = 0          # UNUSED
beta_2 = 1          # UNUSED
data_dim = 28       # UNUSED

# batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

# Read the Input data
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

file_path = "energy_data.csv"
energy_df = pd.read_csv(file_path)

try:
    energy_df = energy_df.set_index('Date').sort_index()
except:
    energy_df = energy_df

# Data transformations to be applied prior to be used with the synthesizer model
energy_data = real_data_loading(energy_df.values, seq_len=seq_len)

print(len(energy_data), energy_data[0].shape)

#Training the TimeGAN synthetizer
if path.exists('synth_energy.pkl'):
    synth = TimeGAN.load('synth_energy.pkl')
else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(energy_data, train_steps=500)
    synth.save('synth_energy.pkl') #save trained model
    
#Generating Synthetic Energy Data
print("Generating new samples ...")
synth_data = synth.sample(len(energy_data))
print(synth_data.shape)