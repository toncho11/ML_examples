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
import gc

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

'''
https://towardsdatascience.com/dont-overfit-how-to-prevent-overfitting-in-your-deep-learning-models-63274e552323
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
'''

import warnings
#warnings.filterwarnings("ignore")

# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions


# get the data from subject of interest
#subject = dataset.subject_list[0]
#raw = dataset._get_single_subject_data(subject)

epochs_all_subjects = [];
label_all_subjects = [];

test_epochs_all_subjects = [];
test_label_all_subjects = [];

print("Train data:")

for subject in dataset.subject_list[0:17]: #[0:17]
    
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
    
    #process raw epochs for the selected subject 
    for i in range(0, len(epochs_subject)):
        
        single_epoch_subject_data = epochs_subject[i]._data[0,:,:]

        #create recurrence plot of a single epoch
        rp = RecurrencePlot(threshold='point', percentage=20)
        single_epoch_subject_rp = rp.fit_transform(single_epoch_subject_data)
        print(single_epoch_subject_rp.shape)
    
        #add to list
        epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1
        
        del single_epoch_subject_data
        del rp
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()


print("Test data:================================================================================================================")

for subject in dataset.subject_list[17:]: #[17:]
    
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
    
    #process raw epochs for the selected subject 
    for i in range(0, len(epochs_subject)):
        
        single_epoch_subject_data = epochs_subject[i]._data[0,:,:]

        #create recurrence plot of a single epoch
        rp = RecurrencePlot(threshold='point', percentage=20)
        single_epoch_subject_rp = rp.fit_transform(single_epoch_subject_data)
        print(single_epoch_subject_rp.shape)
    
        #add to list
        test_epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        test_label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1
        
        del single_epoch_subject_data
        del rp
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()

#train_images = np.array(epochs_all_subjects)
#train_images = np.array(epochs_all_subjects).reshape(170,769,769,1)
train_images = np.array(epochs_all_subjects)[:, :, :, np.newaxis] # we add an extra axis as required by keras
train_labels = np.array(label_all_subjects)

#train_images = tf.expand_dims(train_images, axis=3).shape.as_list()

test_images = np.array(test_epochs_all_subjects)[:, :, :, np.newaxis]
test_labels = np.array(test_label_all_subjects)


img_size = 769#train_images[0].shape[0]

#build model

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(img_size,img_size,1)))
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

#model.fit(train_images, train_labels, epochs=8)
#model.fit(train_images, train_labels, epochs=8, validation_split=0.2, shuffle=True)
model.fit(train_images, train_labels, epochs=20, shuffle=True)

#training results
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy on unseen data:', test_acc)

print("Done.")