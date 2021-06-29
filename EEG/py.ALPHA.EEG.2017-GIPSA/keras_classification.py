import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from alphawaves.dataset import AlphaWaves

import mne

"""
=============================
Example on how to classify using Deep Learning with Keras
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2

"""
# Authors: Anton Andreev
#
# License: BSD (3-clause)

import warnings
#warnings.filterwarnings("ignore")

# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions


# get the data from subject of interest
subject = dataset.subject_list[0]
raw = dataset._get_single_subject_data(subject)

# filter data and resample
fmin = 3
fmax = 40
raw.filter(fmin, fmax, verbose=False)
raw.resample(sfreq=128, verbose=False)

# detect the events and cut the signal into epochs
events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
event_id = {'closed': 1, 'open': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False, preload=True)
epochs.pick_types(eeg=True)

# parameters 
nb_train_samples = 400 
nb_validation_samples = 100
epochs = 10
batch_size = 16

train_generator = [] #todo
validation_generator = [] #todo
#create model

input_shape = (3, img_width, img_height) #todo  

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
  
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# actual traininig
model.fit(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs, 
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

#loss, acc = model.evaluate(x_test,  y_test, verbose=2) #todo

print("Done.")