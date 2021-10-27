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

import mne
from pyts.image import RecurrencePlot
import gc

import os
import glob
import sys
sys.path.append('.')
from braininvaders2014a.dataset import BrainInvaders2014a
from pathlib import Path

def multivariateRP(sample, electrodes, dimension, time_delay, percentage):
    
    channels_N = sample.shape[0]
    
    #Time window = T
    #delta = 40, the interval T is chpped into epochs of delta elements 
    #T is the time interval to be taken from the epoch sample beginning
       
    delta = time_delay 
    points_n = dimension

    #we need to leave enough space for the last space at the end to perform n=dimension jumps over time_delay data
    T = sample.shape[1] - ((dimension-1) * time_delay)
    
    #T = ((dimension-1) * time_delay)
     
    print("T=",T)
    X_traj = np.zeros((T,points_n * channels_N))
            
    for i in range(0,T): #delta is number of vectors with  length points_n
        
        for j in range(0,points_n): #j is inside the sample data and jumps over the points
            start_pos = j * delta
            pos = start_pos + i
            
            for e in electrodes:
                #print(e)
                pos_e = (e * points_n) + j
                #print(pos_e)
                #all points first channel,
                print(i, pos_e, e, pos)
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

dataset = BrainInvaders2014a() #https://hal.archives-ouvertes.fr/hal-02171575/document

def CreateData(m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject):
    
    folder = "D:\Work\ML_examples\EEG\py.BI.EEG.2014a-GIPSA\data"
    
    folder = folder + "\\rp_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) 
    
    # print(folder)
    
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    # print("Clean data:")
    
    # files = glob.glob(folder + "\\*")
    # for f in files:
    #     if f.endswith(".npy"):
    #         os.remove(f)
        
    print("Write rp image data:")
    
    
    for subject in range(1,2):
    
        #load data
        print("Subject =",subject)
        sessions = dataset._get_single_subject_data(subject)
        raw = sessions['session_1']['run_1']
    
        # filter data and resample
        fmin = filter_fmin
        fmax = filter_fmax
        raw.filter(fmin, fmax, verbose=False)
    
        # detect the events and cut the signal into epochs
        events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
        event_id = {'NonTarget': 1, 'Target': 2}
        epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
        epochs.pick_types(eeg=True)
    
        # get trials and labels
        
        epochs_subject = epochs
        
        epochs_class_1 = 0
        epochs_class_2 = 0
        
        for i in range(0, 1): 
            
            single_epoch_subject_data = epochs_subject[i]._data[0,:,:]
    
            label = list(epochs_subject[i].event_id.values())[0]-1 #sigmoid requires that labels are [0..1]
            
            #save
            if (label==0 and epochs_class_1 < max_epochs_per_subject) or (label==1 and epochs_class_2 < max_epochs_per_subject):
    
                single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, percentage)
                
                plt.imshow(single_epoch_subject_rp, cmap='binary', origin='lower')

#m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject
                    
CreateData(4,30,1,20,[6,13,14,15],10,20,20)