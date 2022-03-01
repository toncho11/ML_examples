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
import sys

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
def ShowPerDataset(dataset, channel, show_plot):
    
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
    if (count == 0):
        print("Error: no files detected!")
        sys.exit()
    
    length = len(np.load(folder + files[0]))
    summTarget = np.zeros(length)
    summNonTarget = np.zeros(length)
    
    for f in files:
        if f.find("_ch_" + str(channel)) != -1:
            if f.endswith("_NonTarget.npy"):
                summNonTarget = summNonTarget + np.load(folder + f)
            elif f.endswith("_Target.npy"):
                summTarget = summTarget + np.load(folder + f)
    
    if (sum(summTarget) == 0 or sum(summNonTarget) == 0):
        print("Error: no files for this channel")
        sys.exit()
        
    averageNonTarget = summNonTarget / count
    averagetarget = summTarget / count
    
    if show_plot:
        plt.plot(averageNonTarget, "-b", label="Non Target")
        plt.plot(averagetarget, "-r", label="Target")
        plt.legend(loc="upper left")
    else:
        return averagetarget, averageNonTarget

def PlotDataSets():
    fig, axs = plt.subplots(2,2) #rows, columns
    fig.set_size_inches(18.5, 10.5)
    plt.rc('font', size=8)
    plt.rc('axes', titlesize=8)
    
    #["Cz", "Pz", "Oz"]
    ch = "Oz"
    dataset = "BNCI2014008"
    averagetarget, averageNonTarget = ShowPerDataset(dataset, GetElectrodeByName(dataset, ch), False)
    axs[0,0].set_title(dataset)
    #axs[0,0].plot(averageNonTarget, "-b", label="Non Target")
    axs[0,0].plot(averagetarget, "-r", label="Target")
    axs[0,0].legend(loc="upper left")
    
    dataset = "bi2013a"
    averagetarget, averageNonTarget = ShowPerDataset(dataset, GetElectrodeByName(dataset, ch), False)
    axs[0,1].set_title(dataset)
    #axs[0,1].plot(averageNonTarget, "-b", label="Non Target")
    axs[0,1].plot(averagetarget, "-r", label="Target")
    axs[0,1].legend(loc="upper left")
    
    dataset = "BNCI2015003"
    averagetarget, averageNonTarget = ShowPerDataset(dataset, GetElectrodeByName(dataset, ch), False)
    axs[1,0].set_title(dataset)
    #axs[1,0].plot(averageNonTarget, "-b", label="Non Target")
    axs[1,0].plot(averagetarget, "-r", label="Target")
    axs[1,0].legend(loc="upper left")
    
    dataset = "BNCI2014009"
    averagetarget, averageNonTarget = ShowPerDataset(dataset, GetElectrodeByName(dataset, ch), False)
    axs[1,1].set_title(dataset)
    #axs[1,1].plot(averageNonTarget, "-b", label="Non Target")
    axs[1,1].plot(averagetarget, "-r", label="Target")
    axs[1,1].legend(loc="upper left")
    
    
#main block - select what you need
subject_show = 6
channel = 14
dataset = "BNCI2014009"

PlotDataSets()

#ShowPerSubjectDataset(dataset, ElectrodeByName(dataset,"CZ"), subject_show)
#ShowPerSubjectDataset(dataset, 1, subject_show)
#ShowPerDataset(dataset, ElectrodeByName(dataset,"CZ"),True)






 
 
 
 