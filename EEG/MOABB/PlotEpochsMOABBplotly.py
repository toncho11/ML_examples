# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:19:20 2022

@author: antona

1) Shows how to extract information such as sampling frequency and channel names from MOABB datasets.

2) Shows how to visualize epochs from a MOABB dataset using Plotly library.

"""

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, DemonsP300 
from moabb.paradigms import P300
import mne

import math
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def PlotEpoch(epoch):
    
    epoch = epoch.transpose(1,0)
    
    df = pd.DataFrame(data = epoch)
    
    for i in range(0, len(df.columns), 1):
        df.iloc[:, i]+= i * 30

    fig = px.line(df, width=600, height=400)
    
    fig.show(renderer='browser')

def PlotEpochs(epochs):
    
    df = pd.DataFrame()
    
    for i in range(0,epochs.shape[0]):
        epoch = epochs[i,:,:]
        epoch = epoch.transpose(1,0)
        df_current = pd.DataFrame(data = epoch, columns=[*range(0, epoch.shape[1], 1)])
        
        for s in range(0, len(df_current.columns), 1):
            df_current.iloc[:, s]+= s * 30
        
        df_current['Source'] = i
        
        if (df.empty):
            df = df_current.copy()
        else: 
            df = pd.concat([df, df_current])
    
    fig = px.line(df, facet_col='Source', facet_col_wrap=4)
    
    fig.show(renderer='browser')
    
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
    
    X_epochs, y, metadata = paradigm.get_data(dataset=d, subjects=[d.subject_list[0]])
    
    PlotEpochs(X_epochs[0:20,:,:])
    print("======================================================================================")