"""
Created on Sun Aug 22 19:54:37 2021

@author: anton andreev
"""

import numpy as np
import matplotlib.pyplot as plt

from alphawaves.dataset import AlphaWaves

import mne
from pyts.image import RecurrencePlot
import gc

#import Image
#import ImageChops
from skimage.metrics import structural_similarity as ssim
from sklearn.neural_network import MLPClassifier
import math
from sklearn import svm
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
#electrode = 14 #get the Oz:14
#electrode = 5 #get the T7:5
#m = 5
#tau = 30 
m = 5 
tau = 40
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
n_train_subjects = 1 #max=19
filter_fmin = 3 #default 3
filter_fmax = 40 #default 40

epochs_all_subjects = [];
labels = [];
electrodes = [];
centroids = np.empty(16, dtype=object) 

test_epochs_all_subjects = [];
test_label_all_subjects = [];

def calcDist(i1, i2):
    return np.sum((i1-i2)**2)

def calcDistManhattan(i1, i2):
    return np.sum(abs((i1-i2)))

def calcDistSSIM(i1, i2):
    return -1 * ssim(i1,i2)

def calculateDistance(i1, i2):
    return calcDist(i1, i2)

channels_N = -1

subjects = dataset.subject_list[0:1]
#sf_subjects = random.sample(subjects, len(subjects))
#train_subjects = sf_subjects[0:n_train_subjects]
#validate_subjects =  sf_subjects[n_train_subjects:]


print("Train data:")

#a = np.zeros((n_train_subjects * 10, 1, 16, 649, 649))

for subject in subjects: #[0:17]
    
    raw = dataset._get_single_subject_data(subject)
    
    # filter data and resample
    raw.filter(filter_fmin, filter_fmax, verbose=False)
    raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs_subject = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False, preload=True)
    epochs_subject.pick_types(eeg=True)
    
    channels_N = len(raw.ch_names)-1
    
    #process raw epochs for the selected subject 
    epochs_n =  2 #len(epochs_subject)
    
    for i in range(1, epochs_n):
        
        #processing a single sample for a subject
        
        #rp = np.zeros((3,5))
        
        sample = epochs_subject[i]._data[0,:,:]    
        X = sample[0,:] #get first electrode
        #B = np.array([X])
        #single_epoch_subject_rp = rp.fit_transform(B)
        
        #Time window = T
        #delta = 40, the interval T is chpped into epochs of delta elements 
        #T is the time interval to be taken from the epoch sample beginning
       
        delta = tau 
        points_n = m
        print(points_n)
        percentage = 20
        T = len(X) - ((m-1) * tau)
         
        X_traj = np.zeros((T,points_n * channels_N))
                
        for i in range(0,T): #delta is number of vectors with  length points_n
            
            for j in range(0,points_n):
                start_pos = j * delta
                pos = start_pos + i
                
                for e in range(0,channels_N):
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

plt.imshow(X_rp, cmap='binary', origin='lower')
#plt.imshow(single_epoch_subject_rp[0,:,:], cmap='binary', origin='lower')