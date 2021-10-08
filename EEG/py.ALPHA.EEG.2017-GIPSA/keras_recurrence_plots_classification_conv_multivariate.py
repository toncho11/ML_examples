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

#Parameters
#10-20 international system
#'Fp1','Fp2','Fc5','Fz','Fc6','T7','Cz','T8','P7','P3','Pz','P4','P8','O1','Oz','O2','stim'
#alpha is at the back of the brain
#start form 0
#electrode = 14 #get the Oz:14
#electrode = 5 #get the T7:5
#m = 5
#tau = 30 
m = 5 
tau = 30
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
n_train_subjects = 4 #max=19
length_s = 5
filter_fmin = 3 #default 3
filter_fmax = 40 #default 40
#electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
electrodes = list(range(0,16))

#sample: rows are channels, columns are the timestamps
def multivariateRP(sample, electrodes, dimension, time_delay, percentage):
    
    channels_N = sample.shape[0]
    
    #Time window = T
    #delta = 40, the interval T is chpped into epochs of delta elements 
    #T is the time interval to be taken from the epoch sample beginning
       
    delta = time_delay 
    points_n = dimension
    print(points_n)
    percentage = 20
    T = sample.shape[1] - ((dimension-1) * time_delay)
     
    X_traj = np.zeros((T,points_n * channels_N))
            
    for i in range(0,T): #delta is number of vectors with  length points_n
        
        for j in range(0,points_n):
            start_pos = j * delta
            pos = start_pos + i
            
            for e in electrodes:
                #print(e)
                pos_e = (e * points_n) + j
                #print(pos_e)
                #all points first channel, 
                X_traj[i, pos_e ] = sample[e,pos] #i is the vector, j is indexing isnide the vector 
            #print(pos)
            
    X_dist = np.zeros((T,T))
    
    #calculate distances
    for i in range(0,T): #i is the vector
        for j in range(0,T):
             v1 = X_traj[i,:]
             v2 = X_traj[j,:]
             X_dist[i,j] = np.sqrt( np.sum((v1 - v2) ** 2) ) 
    
    percents = np.percentile(X_dist,percentage)
    
    X_rp = X_dist < percents
    
    return X_rp


train_epochs_all_subjects = [];
train_label_all_subjects = [];

test_epochs_all_subjects = [];
test_label_all_subjects = [];

print("Train data:")

for subject in dataset.subject_list[0:n_train_subjects]: #get train data
    
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

        label = list(epochs_subject[i].event_id.values())[0]-1
        #create recurrence plot of a single epoch
        # rp = RecurrencePlot(threshold='point', percentage=20)
        # single_epoch_subject_rp = rp.fit_transform(single_epoch_subject_data)
        # print(single_epoch_subject_rp.shape)
    
        # #add to list
        # epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        # label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1
        
          
        single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, 20)
        train_epochs_all_subjects.append(single_epoch_subject_rp.copy())
        train_label_all_subjects.append(label)
            
        del single_epoch_subject_data
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()


print("Test data:================================================================================================================")

for subject in dataset.subject_list[n_train_subjects:length_s]: #get test data
    
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

        label = list(epochs_subject[i].event_id.values())[0] - 1 
        
        single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, 20)
        test_epochs_all_subjects.append(single_epoch_subject_rp.copy())
        test_label_all_subjects.append(label)
        
        del single_epoch_subject_data
        del single_epoch_subject_rp    
        gc.collect();
    
    del raw
    del epochs_subject
    gc.collect()

#train_images = np.array(epochs_all_subjects)
#train_images = np.array(epochs_all_subjects).reshape(170,769,769,1)
train_images = np.array(train_epochs_all_subjects)[:, :, :, np.newaxis] # we add an extra axis as required by keras
train_labels = np.array(train_label_all_subjects)

#train_images = tf.expand_dims(train_images, axis=3).shape.as_list()

test_images = np.array(test_epochs_all_subjects)[:, :, :, np.newaxis]
test_labels = np.array(test_label_all_subjects)


img_size = train_images[0].shape[1]
print(img_size)

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