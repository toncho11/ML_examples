# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:27:00 2022

@author: antona

Performing data augmentation on the P300 class in an EEG dataset and classification.

"""

import os
import glob
import time
import sys
import numpy as np

#skleran
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import sklearn.model_selection
import sklearn.datasets
#import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

#moabb
from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300
from moabb.evaluations import WithinSessionEvaluation

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

#TimeVAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
sys.path.insert(1, 'C:/Work/PythonCode/ML_examples/AutoEncoders/TimeVAE') #should be changed to your TimeVAE code
#from local files:
import utils
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE

#PyRiemann
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

#START CODE

def ToOneSubject(db): 
    
    #data should be suffled very well
    
    return db

class DataAugment(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='scm'):#FIX: one needs to select which class to augment
        """Init."""
        self.estimator = estimator 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        #Final sampling from the vae
        #num_samples = 100
        #new_samples = vae.get_prior_samples(num_samples=num_samples)
        
        # inverse-transform scaling 
        #new_samples = scaler.inverse_transform(new_samples)
        return X

pipelines = {}
#XdawnCovariances can be used

#should test with other than MDM pipelines!
pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result
pipelines["DataAugment+MDM"] = make_pipeline(DataAugment(), Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result

print("Total pipelines to evaluate: ", len(pipelines))

datasets = [BNCI2014008()]

datasets = [ToOneSubject(db) for db in datasets]

#apply ToOneSubject to all datasets

subj = [1, 2, 3]

for d in datasets:
    d.subject_list = subj

paradigm = P300()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)

results = evaluation.process(pipelines)

print(results.groupby('pipeline').mean('score')[['score', 'time']])