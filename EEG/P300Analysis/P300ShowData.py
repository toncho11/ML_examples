# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:56:34 2022

@author: Anton Andreev
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from os import walk

from DatasetHelper import *

#config
base_folder = "h:" + os.path.sep + "data" + os.path.sep

#show average for a single channel and single subject
def ShowPerSubjectDataset(dataset, channel, subject_show):

    if (subject_show >= GetSubjectsCount(dataset)):
        print("Error: Non exisitng subject")
        return
    
    if (channel >= GetElectrodeCount(dataset)):
        print("Error: Non exisitng electrode")
        return
    
    #find folder name using dataset name

    folders = glob(os.path.join(base_folder, "*", ""), recursive = False)
    
    #folder = ""
    
    for f in folders:
        if f.endswith(dataset + os.path.sep):
            folder = f
    
    print("Folder = ",folder)
    #see the average per subject for class 0
    classNonTarget = np.load(folder + "subject_" + str(subject_show)+"_ch_" + str(channel) + "_class_NonTarget.npy")
    plt.plot(classNonTarget, "-b", label="Non Target")
 
    #see the average per subject for class 1
    classTarget = np.load(folder + "subject_" + str(subject_show)+"_ch_" + str(channel) + "_class_Target.npy")
    plt.plot(classTarget, "-r", label="Target")
    
    plt.legend(loc="upper left")

#use data from all subjects (that is previously generated) and for a single channel
def ShowPerDataset(dataset, channel):
    
    if (subject_show >= GetSubjectsCount(dataset)):
        print("Error: Non exisitng subject")
        return
    
    if (channel >= GetElectrodeCount(dataset)):
        print("Error: Non exisitng electrode")
        return
    
    #find folder name using dataset name

    folders = glob(os.path.join(base_folder, "*", ""), recursive = False)
    
    #folder = ""
    
    for f in folders:
        if f.endswith(dataset + os.path.sep):
            folder = f
    
    print("Folder = ",folder)
    #read the data for each subject and make the average for class NonTarget
    
    files = next(walk(folder), (None, None, []))[2]
    
    count = len(files)
    length = len(np.load(folder + files[0]))
    summTarget = np.zeros(length)
    summNonTarget = np.zeros(length)
    
    for f in files:
        if f.endswith("_NonTarget.npy"):
            summNonTarget = summNonTarget + np.load(folder + f)
        elif f.endswith("_Target.npy"):
            summTarget = summTarget + np.load(folder + f)
        
    averageNonTarget = summNonTarget / count
    averagetarget = summTarget / count
    
    plt.plot(averageNonTarget, "-b", label="Non Target")
    plt.plot(averagetarget, "-r", label="Target")
    plt.legend(loc="upper left")

#main block - select what you need
subject_show = 6
channel = 14
dataset = "BNCI2014009"

#ShowPerSubjectDataset(dataset, ElectrodeByName(dataset,"CZ"), subject_show)
#ShowPerSubjectDataset(dataset, 1, subject_show)
ShowPerDataset(dataset, ElectrodeByName(dataset,"CZ"))



 
 
 
 