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
from moabb.datasets.base import BaseDataset

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

class OneSubject(BaseDataset):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.subject_list = [0]
        self.n_sessions = 1
        self.event_id = dataset.event_id
        self.code = 'Single' + dataset.code
        self.interval = dataset.interval
        self.paradigm = dataset.paradigm
        self.doi = dataset.doi
        self.unit_factor = dataset.unit_factor
        self.data = self._get_data()

    def _get_data(self):
        data = self.dataset.get_data()
        ret = {}
        ret[1] = {}
        ret[1]['session_0'] = {}
        
        i = 0
        for subject, sessions in data.items():
            for session, runs in sessions.items():
                for run, raw in runs.items():
                    ret[1]['session_0']['run_'+str(i)] = raw
                    i = i + 1 
        return ret
    
    def get_data(self, subjects=None):
        return self.data
    
    def _get_single_subject_data(self, subject):
        pass

    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        pass

class DataAugment(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='scm'):#FIX: one needs to select which class to augment
        """Init."""
        self.estimator = estimator 
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        return X #return the same data for now
    
        #FIX: not the correct format (3360, 8, 257) but should be (3360, 257, 8) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        N, T, D = X.shape #N = number of samples, T = time steps, D = feature dimensions
        print(N, T, D)
        
        np.random.shuffle(X)
        
        # min max scale the data    
        scaler = utils.MinMaxScaler()        
       
        scaled_data = scaler.fit_transform(X)
        
        latent_dim = 8
        
        vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
        
        vae.compile(optimizer=Adam())
        # vae.summary() ; sys.exit()

        early_stop_loss = 'loss'
        #define two callbacks
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
        reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced

        vae.fit(
            scaled_data, 
            batch_size = 32,
            epochs=500,
            shuffle = True,
            callbacks=[early_stop_callback, reduceLR],
            verbose = 1
        )
        
        #Final sampling from the vae
        num_samples = 100 #FIX: set to the correct number we need
        new_samples = vae.get_prior_samples(num_samples=num_samples)
        
        # inverse-transform scaling 
        new_samples = scaler.inverse_transform(new_samples)
        
        #return X #return the same data for now

pipelines = {}
#XdawnCovariances can be used

#should test with other than MDM pipelines!
#pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result
pipelines["DataAugment+MDM"] = make_pipeline(DataAugment(), Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result

print("Total pipelines to evaluate: ", len(pipelines))

datasets = [BNCI2014008()]

#maybe not working correctly:
# subj = [1, 2, 3]
# for d in datasets:
#     d.subject_list = subj

#merge all subjects in one single subject, so that we train on the entire dataset
datasets = [OneSubject(db) for db in datasets]

paradigm = P300()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)

results = evaluation.process(pipelines)

print(results.groupby('pipeline').mean('score')[['score', 'time']])