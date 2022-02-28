# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:58:35 2022

@author: Anton Andreev
"""

#This script ..............................................................

import matplotlib.pyplot as plt
from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300
import numpy as np

from sklearn.preprocessing import LabelEncoder

import os
import glob
import time

from DatasetHelper import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

#datasets = [BNCI2014008()] # , bi2013a(), EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
paradigm = P300()
le = LabelEncoder()

def CreateData(dataset, channels, n_subjects, max_epochs_per_subject):
    
    #folder = "C:\\Work\PythonCode\\ML_examples\\EEG\\moabb.bi2013a\\data"
    #folder = "h:\\data"
    folder = "h:\\data"
    #folder = "c:\\temp\\data"

    folder = folder + "\\p300_sub_" + str(n_subjects) + "_max_epochs_" + str(max_epochs_per_subject) + "_dataset_" + dataset.__class__.__name__ 
    
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write P300 image data:")
    
    
    #for dataset in datasets:
        
    for subject_i, subject in enumerate(dataset.subject_list[0:n_subjects]):
        
        epochs_class_1 = 0
        epochs_class_2 = 0
        
        print("Loading subject:" , subject)  
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        y = le.fit_transform(y)
        print(X.shape) 
        #0 NonTarget
        #1 Target       
        print("All class target samples: ", sum(y))
        print("All class non-target samples: ", len(y) - sum(y))

        index_label1 = [];
        index_label2 = [];
        
        #get only the required number of samples
        for idx,val in enumerate(y):
            if (val == 0 and epochs_class_1 < max_epochs_per_subject):
                index_label1.append(idx)
                epochs_class_1 = epochs_class_1 + 1
            elif (val == 1 and epochs_class_2 < max_epochs_per_subject):
                index_label2.append(idx)
                epochs_class_2 = epochs_class_2 + 1
        
        print("Using class target samples: ", epochs_class_2)
        print("Using class non-target samples: ", epochs_class_1)
        
        for P300_channel in channels:
            imave1 = np.average(X[index_label1][P300_channel],axis=0)
            imave2 = np.average(X[index_label2][P300_channel],axis=0)
            
            filename = "subject_" + str(subject_i) + "_ch_" + str(P300_channel) + '_class_NonTarget'
            full_filename = folder + "\\" + filename
            print("Saving: " + full_filename)
            #plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
            np.save(full_filename, imave1)
            
            filename = "subject_" + str(subject_i) + "_ch_" + str(P300_channel) + "_class_Target"
            full_filename = folder + "\\" + filename
            print("Saving: " + full_filename)
            #plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
            np.save(full_filename, imave2)
        
        

if __name__ == '__main__':

    start = time.time()
    f1 = paradigm.filters[0][0]
    f2 = paradigm.filters[0][1]
 
    channel = 6
    sub_max = 30
    d = BNCI2014009()
    
    CreateData(d, GetChannelRangeInt(GetDatasetNameAsString(d)), sub_max, 400)
    
    #the idea is to have data from both MNE and Zenodo
    #to compare the average P300 for all datasets (4 MNE + 4 Zenodo) and display it
    #to  generate datasets that are correct
    
    end = time.time()
    print("Elapsed time (in seconds):",end - start)
    
    
