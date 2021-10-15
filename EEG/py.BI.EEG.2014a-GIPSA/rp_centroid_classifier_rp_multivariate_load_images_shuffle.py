# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 19:54:37 2021

@author: anton andreev
"""

import numpy as np
import matplotlib.pyplot as plt

from braininvaders2014a.dataset import BrainInvaders2014a

import mne
from pyts.image import RecurrencePlot
import gc

#import Image
#import ImageChops
from skimage.metrics import structural_similarity as ssim
import os
import random
from sklearn.model_selection import train_test_split

"""
=============================
Classification of EGG signal from two states: eyes open and eyes closed.
Here we use centroid classification based on reccurence plots of many electrodes.
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
#dataset = BrainInvaders2014a() # use useMontagePosition = False with recent mne versions


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
tau = 30
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#n_train_subjects = 21 #max=19
#filter_fmin = 4 #default 3
#filter_fmax = 13 #default 40
#electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
#electrodes = list(range(0,16))
epochs_all_subjects = [];
label_all_subjects = [];

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


def ProcessFolder(folder, n_max_subjects):

    train_epochs_all_subjects = [];
    train_label_all_subjects = [];
    
    print("Train data:")
    
    images_loaded = 0
    for filename in os.listdir(folder):
        if filename.endswith(".npy"): 
            
            #print(os.path.join(folder, filename))
            base_name = os.path.basename(filename)
            
            parts = base_name.split("_")
            #print(parts)
            label = int(parts[4].split(".")[0])
            subject = int(parts[1])
            #print("Subject: ", subject, " Label: ", label)
            
            if (subject < n_max_subjects):
                images_loaded = images_loaded + 1
                rp_image=np.load(os.path.join(folder, filename))
                
                train_epochs_all_subjects.append(rp_image)
                
                train_label_all_subjects.append(label + 1)
            
        else:
            continue
    
    print("Train images loaded: ", images_loaded)
    
    train_images1 = []
    train_images2 = []
    
    #separate classes
    for i in range(0,len(train_label_all_subjects)):
        if train_label_all_subjects[i] == 1: # 0 eyes closed = alpha
            train_images1.append(train_epochs_all_subjects[i])#[100:200, 100:250]
        else:
            train_images2.append(train_epochs_all_subjects[i])#[100:200, 100:250]
            
    print("Process Class 1 for Train data:")
    
    images1 = np.array(train_images1) #[:, :, :]
    
    #====================================================================================
    
    print("Process CLass 2 for Train data:")
    
    
    images2 = np.array(train_images2) #[:, :, :]
    #====================================================================================
    
    #reduce images (it seems the performance is the same)
    #images1 = images1[:,330:370,330:370]
    #images2 = images2[:,330:370,330:370]
    
    # ====================================================================================
    # start classification
    
    iterations = 40
    #average_train_accuracy = 0;
    average_classification = 0;
    
    # build centroids
    imave1 = np.average(train_images1,axis=0) #eyes closed, alpha high
    imave2 = np.average(train_images2,axis=0) #eyes opened, alpha low
        
    nn = len(images1)
    all_images = np.concatenate((images1, images2), axis=0)
    labels =  np.concatenate((np.zeros(nn)+1, np.ones(nn)+1), axis=0) 
    print(all_images.shape)
        
    for i in range(iterations):
        
        #shuffle1
        # c = list(zip(all_images, labels))
        # random.shuffle(c)
        # all_images_shuffled, labels_shuffled = zip(*c)
        
        #shuffle2
        indices = np.arange(all_images.shape[0])
        np.random.shuffle(indices)
        all_images_shuffled = all_images[indices]
        labels_shuffled = labels[indices]
        
        #plt.imshow(imave1, cmap='binary', origin='lower') #NON TARGET
        #plt.imshow(imave2, cmap='binary', origin='lower') #TARGET
        
        X_train, X_test, y_train, y_test = train_test_split(all_images_shuffled, labels_shuffled, test_size=0.4)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # validation test ==========================================================================
        correctly_classified = 0;
        valid_all_N = len(X_test)
        
        #print("Class 0 eyes closed")
        for i in range(0,len(X_test)):
            
            x = X_test[i]
            y = y_test[i]
            
            d1 = calculateDistance(x,imave1) #alpha 0
            
            d2 = calculateDistance(x,imave2) #non alpha 1
            
            #print("Data: ", d1, d2)
            
            if (d1 < d2 and y == 1):
                #print("True")
                correctly_classified = correctly_classified + 1
            else:
                pass
                #print("False")
                
            if (d2 < d1 and y == 2):
                #print("True")
                correctly_classified = correctly_classified + 1
            else:
                pass
                #print("False")
            
                
        #print("Cross correlation: ", correctly_classified / valid_all_N)
        average_classification = average_classification + correctly_classified / valid_all_N
    
    #print("======================================================================================")
    #print("Train average accuracy: ", average_train_accuracy / iterations)
    print("Average cross classification: ", average_classification / iterations)
    
    return average_classification / iterations

# results = []
data_folder="D:\\Work\\ML_examples\\EEG\\py.BI.EEG.2014a-GIPSA\\data"
# for x in os.walk(data_folder):
#     target_folder = x[0]
#     if (target_folder != data_folder):
#         print("target_folder =",target_folder)
#         score = ProcessFolder(target_folder, 100)
#         print("======================================================================================")
#         r = [target_folder,score]
#         results.append(r)

#rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_15_nepo_20', 0.7246875000000002], 
#rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_20_nepo_20', 0.72734375], 
#rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_20_nepo_60', 0.6458854166666665], 
#rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_25_nepo_20', 0.7320312500000001], 

#for i in range(1,1):
ProcessFolder(data_folder + "\\rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_10_nepo_20",100)
ProcessFolder(data_folder + "\\rp_m_3_tau_30_f1_1_f2_20_el_4_nsub_10_per_5_nepo_20",100)















