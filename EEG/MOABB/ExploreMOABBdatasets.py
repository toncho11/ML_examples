# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:19:20 2022

@author: antona

1) Shows how to extract information such as sampling frequency and channel names from MOABB datasets.

2) Shows how to visualize epochs from a MOABB dataset using MNE. Epochs are show one after another separated by a vertical line.

For the interactive plot backend 'qt' the package 'mne-qt-browser' is required.
Install with: pip install mne matplotlib mne-qt-browser

More on mne.viz.plot_epochs: https://mne.tools/stable/generated/mne.viz.plot_epochs.html#

"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300
import mne

def PlotEpochs(ds):
    #plots
    #run1.plot()
    #mne.viz.plot_raw(run1)
    
    #currently the first subject is selected
    X_epochs, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]], return_epochs=True)
    #mne.viz.plot_epochs_image(X_epochs)
    
    #select backend
    #mne.viz.set_browser_backend('matplotlib',verbose=None)
    mne.viz.set_browser_backend('qt',verbose=None) #requires: mne-qt-browser package
    
    #it shows all epochs one after another
    mne.viz.plot_epochs(X_epochs, scalings='auto')
    
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
#for d in [ BNCI2014009()]:
for d in [ bi2014a()]:
#for d in [bi2014a(), bi2014b(), bi2015a(), bi2015b(), BNCI2014008(), BNCI2014009(), BNCI2015003(), EPFLP300(), DemonsP300()]:
    GetDataSetInfo(d)
    PlotEpochs(d)
    print("======================================================================================")