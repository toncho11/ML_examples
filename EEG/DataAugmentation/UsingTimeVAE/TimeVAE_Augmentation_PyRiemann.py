# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:56:26 2022

@author: antona
"""

# import matplotlib.pyplot as plt
# from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
# from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

import numpy as np

from sklearn.preprocessing import LabelEncoder

#import Dither #pip install PyDither
import os
import glob
import time
import sys

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

#import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
sys.path.insert(1, 'C:/Work/PythonCode/ML_examples/AutoEncoders/TimeVAE') #should be changed to your TimeVAE code
#from local files:
import utils
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE

#start code

paradigm = P300()

le = LabelEncoder()

# Puts all subjects in single X,y
def BuidlDataset(dataset):
    
    subjects  = enumerate(dataset.subject_list)
    
    X = np. array([])
    y = np. array([])
    
    for subject_i, subject in subjects:
        
        #epochs_class_1 = 0
        #epochs_class_2 = 0
        
        print("Loading subject:" , subject) 
        
        X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        
        y1 = le.fit_transform(y1)
        print(X1.shape)  
        
        #0 NonTarget
        #1 Target       
        print("Total class target samples available: ", sum(y1))
        print("Total class non-target samples available: ", len(y1) - sum(y1))
        
        if (X.size == 0):
            X = np.copy(X1)
            y = np.copy(y1)
        else:
            X = np.concatenate((X, X1), axis=0)
            y = np.concatenate((y, y1), axis=0)
    
    print("Building train data completed: ", X.shape)
    return X,y

# should be changed to be K-fold
def EvaluatePyRiemann(X_train, X_test, y_train, y_test):
    pass

# Augments the p300 class with TimeVAE
def AugmentData(X_p300):
    pass
    
if __name__ == "__main__":
    
    #select dataset to be used
    db = BNCI2014008()
    
    X, y = BuidlDataset(db)
    
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)
        
    print('Test with PyRiemann, NO data augmentation')
    #This produces a base result to compare with
    EvaluatePyRiemann(X_train, X_test, y_train, y_test)
    
    #Perform data augmentation with TimeVAE
    
    X_augmented = AugmentData(X_train)
    
    #add to X_train and y_train
    
    #shuffle the real training data and the augmented before testing again
    
    print('Test with PyRiemann, WITH data augmentation')
    EvaluatePyRiemann(X_train, X_test, y_train, y_test)