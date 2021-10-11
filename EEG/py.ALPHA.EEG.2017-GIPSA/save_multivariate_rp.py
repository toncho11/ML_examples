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

import os
import glob

"""
=============================
Saving multivariate rps 
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
#n_train_subjects = 1 #max=19
length_s = 19
filter_fmin = 4 #default 3
filter_fmax = 13 #default 40
electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
#electrodes = list(range(0,16))
folder = "D:\Work\ML_examples\EEG\py.ALPHA.EEG.2017-GIPSA\multivariate_rp_images"
print(folder)

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

print("Clean data:")

files = glob.glob(folder + "\\*")
for f in files:
    if f.endswith(".npy"):
        os.remove(f)
    
print("Train data:")

for subject in dataset.subject_list[0:length_s]: #get train data
    
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
        
        single_epoch_subject_data = epochs_subject[i]._data[0,:,:] #16 x 769

        label = list(epochs_subject[i].event_id.values())[0]-1 #sigmoid requires that labels are [0..1]
          
        single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, 20)
        train_epochs_all_subjects.append(single_epoch_subject_rp.copy())
        train_label_all_subjects.append(label)
            
        filename = "subject_" + str(subject-1) + "_rp_label_" + str(label) + "_epoch_" + str(i)
        full_filename = folder + "\\" + filename
        print("Saving: " + full_filename)
        np.save(full_filename, single_epoch_subject_rp)

#test save is ok
rp_image=np.load(full_filename + ".npy")
plt.imshow(rp_image, cmap='binary', origin='lower')
print("Done.")