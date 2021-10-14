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
dataset = BrainInvaders2014a() #https://hal.archives-ouvertes.fr/hal-02171575/document

#Parameters
#10-20 international system
# Fp1,  Fp2,  F5,  AFZ,  F6,   T7,  Cz,  T8,  P7,  P3 , PZ,  P4 , P8,  O1,  Oz,  O2
#  0     1     2    3    4     5    6    7    8    9    10   11   12   13   14   15
#alpha is at the back of the brain
#start form 0
#electrode = 14 #get the Oz:14
#electrode = 5 #get the T7:5
#m = 5
#tau = 30 
#m = 5 #max 15 
#tau = 40
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#n_train_subjects = 1 #max=19
#length_s = 19 # max 60 !!!!!!!!!
#filter_fmin = 1 #default 3
#filter_fmax = 20 #default 40
#electrodes = [6,13,14,15]
#electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
#electrodes = list(range(0,16))


#sample: rows are channels, columns are the timestamps
def multivariateRP(sample, electrodes, dimension, time_delay, percentage):
    
    channels_N = sample.shape[0]
    
    #Time window = T
    #delta = 40, the interval T is chpped into epochs of delta elements 
    #T is the time interval to be taken from the epoch sample beginning
       
    delta = time_delay 
    points_n = dimension
    print(points_n)
    percentage = 10
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


def CreateData(m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject):
    
    folder = "D:\Work\ML_examples\EEG\py.BI.EEG.2014a-GIPSA\data"
    
    folder = folder + "\\rp_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) 
    
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write rp image data:")
    
    
    for subject in range(1,n_subjects+1):
    
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
        
        for i in range(0, len(epochs)): 
            
            single_epoch_subject_data = epochs_subject[i]._data[0,:,:] #16 x 769
    
            label = list(epochs_subject[i].event_id.values())[0]-1 #sigmoid requires that labels are [0..1]
            
            #save
            if (label==0 and epochs_class_1 < max_epochs_per_subject) or (label==1 and epochs_class_2 < max_epochs_per_subject):
    
                single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, percentage)
                
                filename = "subject_" + str(subject-1) + "_rp_label_" + str(label) + "_epoch_" + str(i)
                full_filename = folder + "\\" + filename
                print("Saving: " + full_filename)
                np.save(full_filename, single_epoch_subject_rp)
                
                if (label==0):
                    epochs_class_1 = epochs_class_1 + 1
                    
                if (label==1):
                    epochs_class_2 = epochs_class_2 + 1

#m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject
                    
CreateData(5,30,1,20,[6,13,14,15],10,20,20)

CreateData(5,30,1,20,[6,13,14,15],10,20,20)
CreateData(8,30,1,20,[6,13,14,15],10,20,20)
CreateData(10,30,1,20,[6,13,14,15],10,20,20)
CreateData(14,30,1,20,[6,13,14,15],10,20,20)

CreateData(5,10,1,20,[6,13,14,15],10,20,20)
CreateData(5,40,1,20,[6,13,14,15],10,20,20)
CreateData(5,50,1,20,[6,13,14,15],10,20,20)

CreateData(8,10,1,20,[6,13,14,15],10,20,20)
CreateData(3,30,1,20,[6,13,14,15],10,20,20)
CreateData(10,40,1,20,[6,13,14,15],10,20,20)

#=====================================================
CreateData(5,30,1,20,list(range(0,16)),10,20,20)
CreateData(8,30,1,20,list(range(0,16)),10,20,20)
CreateData(10,30,1,20,list(range(0,16)),10,20,20)
CreateData(14,30,1,20,list(range(0,16)),10,20,20)

CreateData(5,10,1,20,list(range(0,16)),10,20,20)
CreateData(5,40,1,20,list(range(0,16)),10,20,20)
CreateData(5,50,1,20,list(range(0,16)),10,20,20)

CreateData(8,10,1,20,list(range(0,16)),10,20,20)
CreateData(3,30,1,20,list(range(0,16)),10,20,20)
CreateData(10,40,1,20,list(range(0,16)),10,20,20)

#=====================================================

CreateData(5,30,1,20,[6,13,14,15],10,15,20)

CreateData(5,30,1,20,[6,13,14,15],10,15,20)
CreateData(8,30,1,20,[6,13,14,15],10,15,20)
CreateData(10,30,1,20,[6,13,14,15],10,15,20)
CreateData(14,30,1,20,[6,13,14,15],10,15,20)

CreateData(5,10,1,20,[6,13,14,15],10,15,20)
CreateData(5,40,1,20,[6,13,14,15],10,15,20)
CreateData(5,50,1,20,[6,13,14,15],10,15,20)

CreateData(8,10,1,20,[6,13,14,15],10,15,20)
CreateData(3,30,1,20,[6,13,14,15],10,15,20)
CreateData(10,40,1,20,[6,13,14,15],10,15,20)

#==================================================

CreateData(5,30,1,20,[6,13,14,15],10,25,20)

CreateData(5,30,1,20,[6,13,14,15],10,25,20)
CreateData(8,30,1,20,[6,13,14,15],10,25,20)
CreateData(10,30,1,20,[6,13,14,15],10,25,20)
CreateData(14,30,1,20,[6,13,14,15],10,25,20)

CreateData(5,10,1,20,[6,13,14,15],10,25,20)
CreateData(5,40,1,20,[6,13,14,15],10,25,20)
CreateData(5,50,1,20,[6,13,14,15],10,25,20)

CreateData(8,10,1,20,[6,13,14,15],10,25,20)
CreateData(3,30,1,20,[6,13,14,15],10,25,20)
CreateData(10,40,1,20,[6,13,14,15],10,25,20)

#=====================================================

CreateData(5,30,1,20,[6,13,14,15],10,20,60)

CreateData(5,30,1,20,[6,13,14,15],10,20,60)
CreateData(8,30,1,20,[6,13,14,15],10,20,60)
CreateData(10,30,1,20,[6,13,14,15],10,20,60)
CreateData(14,30,1,20,[6,13,14,15],10,20,60)

CreateData(5,10,1,20,[6,13,14,15],10,20,60)
CreateData(5,40,1,20,[6,13,14,15],10,20,60)
CreateData(5,50,1,20,[6,13,14,15],10,20,60)

CreateData(8,10,1,20,[6,13,14,15],10,20,60)
CreateData(3,30,1,20,[6,13,14,15],10,20,60)
CreateData(10,40,1,20,[6,13,14,15],10,20,60)

#====================================================

CreateData(5,30,1,20,[6,13,14,15],20,20,30)

CreateData(5,30,1,20,[6,13,14,15],20,20,30)
CreateData(8,30,1,20,[6,13,14,15],20,20,30)
CreateData(10,30,1,20,[6,13,14,15],20,20,30)
CreateData(14,30,1,20,[6,13,14,15],20,20,30)

CreateData(5,10,1,20,[6,13,14,15],20,20,30)
CreateData(5,40,1,20,[6,13,14,15],20,20,30)
CreateData(5,50,1,20,[6,13,14,15],20,20,30)

CreateData(8,10,1,20,[6,13,14,15],20,20,30)
CreateData(3,30,1,20,[6,13,14,15],20,20,30)
CreateData(10,40,1,20,[6,13,14,15],20,20,30)

#====================================================
#test save is ok
#rp_image=np.load(full_filename + ".npy")
#plt.imshow(rp_image, cmap='binary', origin='lower')
print("Done.")