# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 19:54:37 2021

@author: anton
"""

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

#Parameters
#10-20 international system
#'Fp1','Fp2','Fc5','Fz','Fc6','T7','Cz','T8','P7','P3','Pz','P4','P8','O1','Oz','O2','stim'
#alpha is at the back of the brain
#start form 0
electrode = 14 #get the Oz:14
#electrode = 5 #get the T7:5
#m = 1
#tau = 1 
m = 2 
tau = 30 
rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
n_train_subjects = 1

epochs_all_subjects = [];
label_all_subjects = [];

test_epochs_all_subjects = [];
test_label_all_subjects = [];

print("Train data:")

for subject in dataset.subject_list[0:n_train_subjects]: #[0:17]
    
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
        X = single_epoch_subject_data[electrode,:]
        X = np.array([X])
        single_epoch_subject_rp = rp.fit_transform(X)
        print(X.shape)
    
        #add to list
        epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1
        
        del single_epoch_subject_data
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()


print("Test data:================================================================================================================")

for subject in dataset.subject_list[n_train_subjects:n_train_subjects+1]: #[17:]
    
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
        X = single_epoch_subject_data[electrode,:]
        X = np.array([X])
        single_epoch_subject_rp = rp.fit_transform(X)
        print(X.shape)
    
        #add to list
        test_epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        test_label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1
        
        del single_epoch_subject_data
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()

del rp

train_images = np.array(epochs_all_subjects)[:, :, :, np.newaxis] # we add an extra axis as required by keras
train_labels = np.array(label_all_subjects)
print("Number of samples (train + validation):", len(train_labels))

test_images = np.array(test_epochs_all_subjects)[:, :, :, np.newaxis]
test_labels = np.array(test_label_all_subjects)


img_size = train_images[0].shape[0]
print("Image size", img_size)

#build model 1
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

#compile model
model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
              metrics=['accuracy'])

#build model 2
# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(img_size, img_size)), #no parameter learning just transforming to 1D array
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(2) # Each node contains a score that indicates the current image belongs to one of the classes
# ])

# model.compile(optimizer='adam', #This is how the model is updated based on the data it sees and its loss function.
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
#               metrics=['accuracy']) #Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.




#model.fit(train_images, train_labels, epochs=8)
model.fit(train_images, train_labels, epochs=20, validation_split=0.2, shuffle=True)
#model.fit(train_images, train_labels, epochs=20, shuffle=True)

#training results
print("Testing on unseen data====================================================================")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy on unseen data:', test_acc)

#print("Done.")