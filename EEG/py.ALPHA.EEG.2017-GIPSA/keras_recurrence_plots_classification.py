import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from alphawaves.dataset import AlphaWaves

import mne
from pyts.image import RecurrencePlot

"""
=============================
Classification of EGG signal from two states: eyes open and eyes closed.
Each sample of the two states is represented as an image (a recurrence plot),
next the images are classified using Deep Learning model.  
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2
Pyts 0.11 (a Python Package for Time Series Classification,exists in Anaconda, provides recurrence plots)

"""
# Authors: Anton Andreev
#
# License: BSD (3-clause)

import warnings
#warnings.filterwarnings("ignore")

# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions


# get the data from subject of interest
#subject = dataset.subject_list[0]
#raw = dataset._get_single_subject_data(subject)

epochs = [];

for subject in dataset.subject_list: 
    raw = dataset._get_single_subject_data(subject)
    
    # filter data and resample
    fmin = 3
    fmax = 40
    raw.filter(fmin, fmax, verbose=False)
    raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs_subject = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False, preload=True)
    epochs_subject.pick_types(eeg=True)
    
    #get raw data 
    
    #create recurrence plot and
    
    #add to list
    #mne.Epochs
    #epochs = epochs + epochs_subject


#simple check of data
if epochs[0]._data.shape[1] != 16:
    print("Error: EEG channels are not 16!")


img_width = epochs[0]._data.shape[2]
img_height = epochs[0]._data.shape[1]

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
    
#epochs are 10 (5 closed eyes, 5 open eyes)

# parameters 
#nb_train_samples = 8 #todo
#nb_validation_samples = 2 #todo
#nb_epochs_keras = 16 #number of iterations on the dataset
#batch_size = 2 #todo

train_generator = [] #todo
validation_generator = [] #todo
#create model

#input_shape = (3, img_width, img_height) #todo  

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
# model.fit(train_generator, 
#     steps_per_epoch = nb_train_samples // batch_size,
#     epochs = nb_epochs_keras, 
#     validation_data = validation_generator,
#     validation_steps = nb_validation_samples // batch_size)

#actual traininig
model.fit(train_generator, 
    steps_per_epoch = 5,
    epochs = 6, 
    validation_data = validation_generator,
    validation_steps = 4)

#loss, acc = model.evaluate(x_test,  y_test, verbose=2) #todo

print("Done.")