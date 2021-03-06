# -*- coding: utf-8 -*-
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
Here we use centroid classification based on reccurence plots and multiple electrodes
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
filter_fmin = 3 #default 3
filter_fmax = 40 #default 40

epochs_all_subjects = [];
labels = [];
electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
#electrodes = list(range(0,16))
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
    T = sample.shape[1] - ((m-1) * tau)
     
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

channels_N = -1

subjects = dataset.subject_list

length_s = len(subjects)
#length_s = 2
n_train_subjects = 16 #max=19 , subjects to participate in train

sf_subjects = random.sample(subjects, length_s)
train_subjects = sf_subjects[0:n_train_subjects]
validate_subjects =  sf_subjects[n_train_subjects:]


print("Train data:")

#a = np.zeros((n_train_subjects * 10, 1, 16, 649, 649))

for subject in train_subjects: #[0:17]
    
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
    for i in range(0, len(epochs_subject)):
        
        #processing a single sample for a subject
        
        sample = epochs_subject[i]._data[0,:,:]    
    
        #add to list
        label = list(epochs_subject[i].event_id.values())[0]
        
        # #create recurrence plot of a single epoch/sample        
        # X = sample[c,:]
        # X = np.array([X])
        # single_epoch_subject_rp = rp.fit_transform(X)
        # #print(single_epoch_subject_rp.shape)
        
        
        # epochs_all_subjects.append(single_epoch_subject_rp[0,:,:].copy())
        # labels.append(label)
        # electrodes.append(c)
        # #del single_epoch_subject_rp  
        
        single_epoch_subject_rp = multivariateRP(sample, electrodes, m, tau, 20)
        epochs_all_subjects.append(single_epoch_subject_rp.copy())
        labels.append(label)
        del single_epoch_subject_rp  
        
        #del sample

        #gc.collect();
    
    #del raw
    #del epochs_subject
    #gc.collect()


#images1 = np.array(epochs_all_subjects)[:, :, :]

print(len(epochs_all_subjects), len(labels))

print("================================================")

print("Creating centroids:")


#centroidsEyesClosed = np.empty(channels_N, dtype=object) # contains an averaged rp image
#centroidsEyesOpened = np.empty(channels_N, dtype=object) # contains an averaged rp image
  
#print("Centroid ",e)
epochs_single_sentroid=[]

#only EYES CLOSED
for l in range(0, len(labels)):
    if (labels[l] == 1): #eyes closed alpha
        epochs_single_sentroid.append(epochs_all_subjects[l])
        
#convert np array
RP_images = np.array(epochs_single_sentroid)[:, :, :]

#calculate the average for each centroid
centroidsEyesClosed = np.average(RP_images,axis=0) # for single lectrode, single class from all subjects 

epochs_single_sentroid=[]

# #only EYES OPENED
# for l in range(0, len(labels)):
#     if (labels[l] == 2): #eyes closed alpha
#         epochs_single_sentroid.append(epochs_all_subjects[l])
        
# #convert np array
# RP_images = np.array(epochs_single_sentroid)[:, :, :]

# #calculate the average for each centroid
# centroidsEyesOpened = np.average(RP_images,axis=0) # for single lectrode, single class from all subjects 
# #plt.imshow(centroidsEyesOpened[e], cmap='binary', origin='lower')
     
# ====================================================================================
# start classification

print("Training using the centroids")

iterations = 1
average_train_accuracy = 0;
average_classification = 0;

for i in range(iterations):
    
    local_train_accuracy = 0;
    #train
    
    print("Iteration: ", i)
    
    train_data_X = []
    train_labels = []
    
    print("calculate the distances between the train images and the centroids")
    for subject in train_subjects:

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
              
        for e in range(0, len(epochs_subject)): #go over all epochs for the selected subject
            
            #sample (10 per subject in this case)
            sample = epochs_subject[e]._data[0,:,:]   
            
            rp_image = multivariateRP(sample, electrodes, m, tau, 20)
                
            distEC = calculateDistance(centroidsEyesClosed, rp_image)
            #distEO = calculateDistance(centroidsEyesOpened, rp_image)

            epoch_label = list(epochs_subject[e].event_id.values())[0]     
            train_labels.append(epoch_label)
            
            #x = [distEC,distEO]
            #x = np.concatenate((distEC,distEO))
            #x = distEO
            x = distEC
            #dist_norm = (x-min(x))/(max(x)-min(x))
            train_data_X.append(x)
    
    train_data_X_np =  np.array(train_data_X)
    train_labels_np = np.array(train_labels)
    
    print("Training...")
    #'lbfgs' sover for smaller datasets 
    #clf = MLPClassifier(hidden_layer_sizes=(32,16,8), max_iter=300, activation = 'relu',solver='adam',random_state=1)
    #clf = MLPClassifier(max_iter=300, activation = 'relu',solver='adam',random_state=1)
    #clf = MLPClassifier(hidden_layer_sizes=(32,16,8), max_iter=300, activation = 'relu',solver='lbfgs',random_state=1)
    clf = MLPClassifier(hidden_layer_sizes=(32,16,8), max_iter=800, activation = 'relu',solver='lbfgs',random_state=1)
    #clf = MLPClassifier(hidden_layer_sizes=(64,32,16,8), max_iter=300, activation = 'relu',solver='lbfgs',random_state=1)
    
    #add dimension
    reshaped_train_data = train_data_X_np.reshape(-1, 1)
    #normalize
    reshaped_train_data = (reshaped_train_data-min(reshaped_train_data))/(max(reshaped_train_data)-min(reshaped_train_data))
    
    clf.fit(reshaped_train_data, train_labels_np)
    
    y_pred = clf.predict(reshaped_train_data)
    #print(train_labels_np - y_pred)
    
    local_train_accuracy =  str(train_labels_np - y_pred).count('0') / len (train_labels_np)
    average_train_accuracy = average_train_accuracy + local_train_accuracy
    print("Centroids + NN accuracy (on training data): ", local_train_accuracy)
    
    #SVM
    clf_svm = svm.SVC()
    clf_svm.fit(reshaped_train_data, train_labels_np)
    y_pred_svm = clf_svm.predict(reshaped_train_data)
    print("SVM classification accuracy (on training data): ", str(train_labels_np - y_pred_svm).count('0') / len (train_labels_np))
    
    #validate 
    
    print("Classification ...")
    
    validate_data_X = []
    validate_labels = []
    
    for subject in validate_subjects : #[0:17]

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
              
        for e in range(0, len(epochs_subject)): #go over all epochs for the selected subject
            
            #sample (10 per subject in this case)
            sample = epochs_subject[e]._data[0,:,:]   
            
            rp_image = multivariateRP(sample, electrodes, m, tau, 20)
                
            distEC = calculateDistance(centroidsEyesClosed, rp_image)
            #distEO = calculateDistance(centroidsEyesOpened, rp_image)
            
            epoch_label = list(epochs_subject[e].event_id.values())[0]
            
            # print("Label ", epoch_label, sum(distEC), sum(distEO))
            # if (epoch_label == 1 and sum(distEO) > sum(distEC)):
            #     print("OK")
            # else:
            #     print("NOT OK")
                
            # if (epoch_label == 2 and sum(distEO) < sum(distEC)):
            #     print("OK")
            # else:
            #     print("NOT OK")
            
            #x = [distEC, distEO]
            validate_labels.append(epoch_label)
            #x = np.concatenate((distEC,distEO))
            #x = distEO
            x = distEC
            #dist_norm = (x-min(x))/(max(x)-min(x))
            validate_data_X.append(x)
    
    validate_data_X_np =  np.array(validate_data_X)
    validate_labels_np = np.array(validate_labels)
    
    #remove dimension
    validate_data_X_np = validate_data_X_np.reshape(-1, 1)
    #normalize
    validate_data_X_np = (validate_data_X_np-min(validate_data_X_np))/(max(validate_data_X_np)-min(validate_data_X_np))
    
    y_validate = clf.predict(validate_data_X_np)
    #print(validate_labels_np - y_validate)
    local_validate_accuracy = str(validate_labels_np - y_validate).count('0') / len (validate_labels_np)
    print("Centroids + NN accuracy (unseen data): ", local_validate_accuracy) 
    average_classification = average_classification + local_validate_accuracy
    
    #SVM
    y_pred_svm = clf_svm.predict(validate_data_X_np)
    print("SVM classification accuracy (unseen data): ", str(validate_labels_np - y_pred_svm).count('0') / len (validate_labels_np))
    

print("Average train accuracy Centroids + NN: ", average_train_accuracy / iterations)  
print("Average classidfication Centroids + NN on unseen data: ", average_classification / iterations)    
print("done")
    

