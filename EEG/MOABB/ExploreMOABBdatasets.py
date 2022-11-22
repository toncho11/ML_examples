# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:19:20 2022

@author: antona

Shows how to extract information such as sampling frequency and channel names from MOABB datasets.

"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300
import mne

def PlotEpochs(ds):
    #plots
    #run1.plot()
    #mne.viz.plot_raw(run1)
    
    X_epochs, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]], return_epochs=True)
    #mne.viz.plot_epochs_image(X_epochs)
    mne.viz.plot_epochs(X_epochs)
    
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
for d in [bi2014a(), bi2014b(), bi2015a(), bi2015b(), BNCI2014008(), BNCI2014009(), BNCI2015003(), EPFLP300(), DemonsP300()]:
    GetDataSetInfo(d)
    PlotEpochs(d)
    print("======================================================================================")