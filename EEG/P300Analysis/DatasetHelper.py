# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:56:34 2022

@author: Anton Andreev
"""

def GetCzChannel(dataset):
    if (dataset == "BNCI2014008"):
        return 1
    elif (dataset == "bi2013a"):
        return 6
    elif (dataset == "BNCI2015003"):
        return 1
    elif (dataset == "BNCI2014009"):
        return 2
    else:
        print("Error: Could not set Cz for this dataset!")
        
def GetElectrodeCount(dataset):
    if (dataset == "BNCI2014008"):
        return len(GetChannelNames(dataset))
    elif (dataset == "bi2013a"): 
        return len(GetChannelNames(dataset))
    elif (dataset == "BNCI2015003"): 
        return len(GetChannelNames(dataset))
    elif (dataset == "BNCI2014009"): 
        return len(GetChannelNames(dataset))
    else:
        print("Error: Could not get the electrode count!")

# which electrodes to use for P300: "An efficient P300-based brain–computer interface for disabled subjects"
# A comparison of classification techniques for the P300 speller: "Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7","PO8"
def GetChannelNames(dataset):
    if (dataset == "BNCI2014008"): #8 electrodes
        return ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7","PO8"]
    elif (dataset == "bi2013a"): #8 electrodes
        return ["FP1", "FP2", "F5", "AFz", "F6", "T7", "Cz", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
    elif (dataset == "BNCI2015003"): #8 electrodes
        return ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8"]
    elif (dataset == "BNCI2014009"): #8 electrodes
        return ["Fz", "FCz", "Cz", "CPz", "Pz", "Oz", "F3", "F4", "C3", "C4", "CP3", "CP4", "P3", "P4", "PO7", "PO8"]
    else:
        print("Error: Could not get the channel names!")
        
def GetSubjectsCount(dataset):
    if (dataset == "BNCI2014008"):
        return 8
    elif (dataset == "bi2013a"): 
        return 24
    elif (dataset == "BNCI2015003"): 
        return 10
    elif (dataset == "BNCI2014009"): 
        return 10
    else:
        print("Error: Could not get subjects count!")

#if "Cz" is provided then it will return an integer for this electrode
def GetElectrodeByName(dataset, electrode_name):   
    i = 0
    for e in GetChannelNames(dataset):
        if e.lower() == electrode_name.lower():
            return i
        else:
            i = i + 1
            
    print("Error: Electrode ", electrode_name , " not found!")
    
def GetChannelRangeInt(dataset):
    return range(0,GetElectrodeCount(dataset))

def GetDatasetNameAsString(dataset): 
    return dataset.__class__.__name__

def GetEpochLength(dataset):
    if (dataset == "BNCI2014008"):
        return 257
    elif (dataset == "bi2013a"):
        return 513
    elif (dataset == "BNCI2015003"):
        return 206
    elif (dataset == "BNCI2014009"):
        return 206
    else:
        print("Error: Could not get the epoch length!")
        
def GetFrequency(dataset): #in Hz
    if (dataset == "BNCI2014008"):
        return 256
    elif (dataset == "bi2013a"):
        return 512
    elif (dataset == "BNCI2015003"):
        return 256
    elif (dataset == "BNCI2014009"):
        return 256
    else:
        print("Error: Could not get the epoch frequency!")

def GetDataSetInfo(ds):
    
    from moabb.paradigms import P300
    paradigm = P300()
    
    #print("Parameters: ", ds.ds.__dict__)
    print("Dataset name: ", ds.__class__.__name__)
    print("Subjects: ", ds.subject_list)
    print("Subjects count: ", len(ds.subject_list))
    
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]])
    
    print("Electrodes count (inferred): ", X.shape[1])
    print("Epoch length (inferred)    : ", X.shape[2])
    #print("Description:    : ", ds.__doc__)
    