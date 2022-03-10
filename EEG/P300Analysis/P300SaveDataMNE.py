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
        #X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        X, y = FilterNautilus(dataset, subject, 800, []) #["Cz", "Pz", "Oz"]
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
        
        for ch in channels:
            imave1 = np.average(X[index_label1][ch],axis=0)
            imave2 = np.average(X[index_label2][ch],axis=0)
            
            filename = "subject_" + str(subject_i) + "_ch_" + str(ch) + '_class_NonTarget'
            full_filename = folder + "\\" + filename
            print("Saving: " + full_filename)
            #plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
            np.save(full_filename, imave1)
            
            filename = "subject_" + str(subject_i) + "_ch_" + str(ch) + "_class_Target"
            full_filename = folder + "\\" + filename
            print("Saving: " + full_filename)
            #plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
            np.save(full_filename, imave2)
        
#generate all data
def GenerateAllData():
       datasets = [BNCI2014008(), bi2013a(), BNCI2015003(), BNCI2014009()]
       
       sub_max = 30
       epochs_max_per_subject = 2000
       
       for d in datasets:
           CreateData(d, GetChannelRangeInt(GetDatasetNameAsString(d)), sub_max, epochs_max_per_subject)
           
def FilterNautilus(dataset, subject, time_ms, electrodes):
    
    ds_str = GetDatasetNameAsString(dataset)
    
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    y = le.fit_transform(y)
    
    #select the electrodes requested
    
    #calculate the length to take from time
    electrodes_num = []
    
    if (electrodes == []):
        electrodes_num = list(range(0,X.shape[1]))
    else:
        for el in electrodes:
            print(el)
            electrodes_num.append(GetElectrodeByName(ds_str, el))
       
        if (len(electrodes) != len(electrodes_num)):
            print("Error: could not select all the electrodes requested!")
    
    electrodes_num.sort()
    
    old_length = GetEpochLength(ds_str) #length of epoch
    freq = GetFrequency(ds_str)
    new_length = int ((time_ms * freq) / 1000 )
    
    if (new_length > old_length):
        print("Error: new length of epoch is incorrect")
    else:
        print("New epoch length: ", new_length, "/", old_length)
    
    X_new = np.zeros((X.shape[0], int(len(electrodes_num)), new_length))
    
    if (X_new.shape[1] != len(electrodes_num)):
        print("Error: could not select eclectrodes")
        
    for i in range(len(X)):
        X_new[i] = X[i, electrodes_num, 0:new_length]
        
    #remove data that is not good (which one) ?????????????????
    
    return X_new, y

if __name__ == '__main__':

    start = time.time()
    f1 = paradigm.filters[0][0]
    f2 = paradigm.filters[0][1]
 
    channel = 6
    sub_max = 30
    epochs_max_per_subject = 400
    
    GenerateAllData()
    #genearte 
    #d = BNCI2014009()    
    #CreateData(d, GetChannelRangeInt(GetDatasetNameAsString(d)), sub_max, epochs_max_per_subject)
    
    #the idea is to have data from both MNE and Zenodo
    #to compare the average P300 for all datasets (4 MNE + 4 Zenodo) and display it
    #to  generate datasets that are correct
    
    #s1 = FilterNautilus(BNCI2014009() , 1, 300, ["Cz", "Pz", "Oz"])
    
    end = time.time()
    print("Elapsed time (in seconds):",end - start)
    
    
