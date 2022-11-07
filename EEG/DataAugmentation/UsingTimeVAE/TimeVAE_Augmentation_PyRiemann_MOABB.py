# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 12:27:00 2022

@author: antona
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300
from moabb.evaluations import WithinSessionEvaluation

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
#import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

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
    return db

class DataAugment(BaseEstimator, TransformerMixin):
    pass

pipelines = {}
#XdawnCovariances can be used
pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result
pipelines["DataAugment+MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result

print("Total pipelines to evaluate: ", len(pipelines))

datasets = [BNCI2014008()]

#apply ToOneSubject to all datasets

subj = [1, 2, 3]

for d in datasets:
    d.subject_list = subj

paradigm = P300()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)

results = evaluation.process(pipelines)

print(results.groupby('pipeline').mean('score')[['score', 'time']])