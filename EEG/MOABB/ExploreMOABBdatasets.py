# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:19:20 2022

@author: antona
"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300

def GetDataSetInfo(ds):
    #print("Parameters: ", ds.__dict__)
    print("Dataset name: ", ds.__class__.__name__)
    #print("Subjects: ", ds.subject_list)
    print("Subjects count: ", len(ds.subject_list))
    
    # load first subject
    subject = ds.get_data([1])
    # subject 1, session 0, run 0
    run1 = list((list((list(subject.values())[0]).values())[0]).values())[0]
    
    print("Channel names: ", run1.ch_names)
    #print("Channel types: ", run1.get_channel_types())
    print("Sampling frequency: ", run1.info['sfreq'])
    
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]])
    
    print("Electrodes count (inferred): ", X.shape[1])
    print("Epoch length (inferred)    : ", X.shape[2])
    #print("Description:    : ", ds.__doc__)

paradigm = P300()

#not available: bi2013a(), Lee2019_ERP()
for d in [DemonsP300(), bi2014a(), bi2014b(), bi2015a(), bi2015b(), BNCI2014008(), BNCI2014009(), BNCI2015003(), EPFLP300()]:
    GetDataSetInfo(d)
    print("======================================================================================")