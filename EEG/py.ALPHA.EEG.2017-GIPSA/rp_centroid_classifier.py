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
from sklearn.utils import shuffle
import random

"""
=============================
Classification of EGG signal from two states: eyes open and eyes closed.
Here we use centroid classification based on reccurence plots and 1 electrode.
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
Pyts 0.11 (a Python Package for Time Series Classification,exists in Anaconda, provides recurrence plots)

"""
# Authors: Anton Andreev
#
# License: BSD (3-clause)


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
m = 5 
tau = 30 
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
n_train_subjects = 19 #max=19

epochs_all_subjects = [];
label_all_subjects = [];

test_epochs_all_subjects = [];
test_label_all_subjects = [];

def calculateDistance(i1, i2):
    return np.sum((i1-i2)**2)

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
    
        #add to list
        label = list(epochs_subject[i].event_id.values())[0] - 1 #from 1..2 to 0..1
        
        if label == 0:
            #create recurrence plot of a single epoch        
            X = single_epoch_subject_data[electrode,:]
            X = np.array([X])
            single_epoch_subject_rp = rp.fit_transform(X)
            print(X.shape)
            epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
            label_all_subjects.append(label)
            del single_epoch_subject_rp    
        
        del single_epoch_subject_data

        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()


images1 = np.array(epochs_all_subjects)[:, :, :]

#====================================================================================
epochs_all_subjects = []
label_all_subjects = []

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
    
        #add to list
        label = list(epochs_subject[i].event_id.values())[0] - 1 #from 1..2 to 0..1
        
        if label == 1:
            #create recurrence plot of a single epoch        
            X = single_epoch_subject_data[electrode,:]
            X = np.array([X])
            single_epoch_subject_rp = rp.fit_transform(X)
            print(X.shape)
            epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
            label_all_subjects.append(label)
            del single_epoch_subject_rp  
        
        del single_epoch_subject_data 
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()


del rp

images2 = np.array(epochs_all_subjects)[:, :, :]
#====================================================================================

# start classification

iterations = 12
average_classification = 0;

for i in range(iterations):
    
    np.random.shuffle(images1) #Multi-dimensional arrays are only shuffled along the first axis
    np.random.shuffle(images2)
    
    N = len(images1)
    N_validation =  N // 5
    
    # class 0
    train_images1 = images1[:N-N_validation]
    valid_images1 = images1[N-N_validation:]
    
    # class 1
    train_images2 = images2[:N-N_validation]
    valid_images2 = images2[N-N_validation:]
    
    # build centroids
    imave1 = np.average(train_images1,axis=0)
    imave2 = np.average(train_images2,axis=0)
    
    #plt.imshow(imave1, cmap='binary', origin='lower')
    #plt.imshow(imave2, cmap='binary', origin='lower')
    
    # for x in images2:
    #     d1 = calculateDistance(x,imave2)
    #     d2 = calculateDistance(x,imave1)
    #     print("Centroid origin class, centroid other class", d1, d2, d1< d2)
        
    # for x in images1:
    #     d1 = calculateDistance(x,imave1)
    #     d2 = calculateDistance(x,imave2)
    #     print("Centroid origin class, centroid other class", d1, d2, d1< d2)
        
    # validation test
    correctly_classified = 0;
    valid_all_N = len(valid_images1) + len(valid_images2)
    
    #print("Class 0 eyes closed")
    for x in valid_images1:
        d1 = calculateDistance(x,imave1)
        d2 = calculateDistance(x,imave2)
        if (d1 < d2 ):
            #print("True")
            correctly_classified = correctly_classified + 1
        else:
            pass
            #print("False")
        
    #print("Class 1 eyes open")            
    for x in valid_images2:
        d1 = calculateDistance(x,imave2)
        d2 = calculateDistance(x,imave1)
        if (d1 < d2 ):
            correctly_classified = correctly_classified + 1
            #print("True")
        else:
            pass
            #print("False")
            
    print("Cross correlation: ", correctly_classified / valid_all_N)
    average_classification = average_classification + correctly_classified / valid_all_N
    
print("Average classification: ", average_classification / iterations)