# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:36:50 2022

@author: antona
"""

import platform
import sys
if (platform.system() != "Linux"):
    print("auto sklearn is only available on Linux")
    sys.exit()

import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

import numpy as np

from sklearn.preprocessing import LabelEncoder

import Dither #pip install PyDither
import os
import glob
import time
import sys

from joblib import Parallel, delayed
from multiprocessing import Process

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

from mne.preprocessing import Xdawn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

paradigm = P300()

le = LabelEncoder()

def BuidlDataset(dataset, electrodes, n_subjects, max_epochs_per_subject, enableXDAWN):
    
    subjects  = enumerate(dataset.subject_list[0:n_subjects])
    
    for subject_i, subject in subjects:
        
        epochs_class_1 = 0
        epochs_class_2 = 0
        
        print("Loading subject:" , subject) 
        
        #X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
    
        #currently xDawn takes into account the entire dataset and not the reduced one that is used for the actual training
        if (enableXDAWN):
            X_epochs, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
            print("Performing XDawn")
            xd = Xdawn(n_components=6) #output channels = 2 * n_components
            X = np.asarray(xd.fit_transform(X_epochs))
            electrodes = range(X.shape[1])
            print("Finished XDawn")
        else:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        
        y = le.fit_transform(y)
        print(X.shape)  
        
        if (electrodes == []):
            electrodes = list(range(0,X.shape[1]))
        elif X.shape[1] < len(electrodes):
            print("Error: electrode list is longer than electrodes in dataset")
            sys.exit(1)  
        
        print("Electrodes selected: ",electrodes)
        #0 NonTarget
        #1 Target       
        print("Total class target samples available: ", sum(y))
        print("Total class non-target samples available: ", len(y) - sum(y))
        
        return X,y
    

    
if __name__ == "__main__":
    
    X, y = BuidlDataset()
    
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
        
    automl = autosklearn.classification.AutoSklearnClassifier()
    
    automl.fit(X_train, y_train)
    
    y_hat = automl.predict(X_test)
    
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))