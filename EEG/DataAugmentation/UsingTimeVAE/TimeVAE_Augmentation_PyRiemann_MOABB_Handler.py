# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:39:12 2023

@author: antona

Performing data augmentation on the P300 class in an EEG dataset and classification.

"""

# def clear_all():
#     """Clears all the variables from the workspace of the spyder application."""
#     gl = globals().copy()
#     for var in gl:
#         if var[0] == '_': continue
#         if 'func' in str(globals()[var]): continue
#         if 'module' in str(globals()[var]): continue

#         del globals()[var]

# clear_all()

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

class Handler():
    
    def __init__(self, array) -> None:
        
        self.array = np.array(array)
        
        if(not self.supports_np_asarray()):
            self.inject_handler_asarray()
        
        if(not self.supports_np_asanyarray()):
            self.inject_handler_asanyarray()
            
        if(not self.supports_np_reshape()):
            self.inject_handler_reshape()
    
    def inject_handler_asarray(self):
        
        def as_array_func(y, dtype=None, order=None, *, like=None):
            
            if type(y) is Handler:
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                y.array = self.asarray0(y.array, dtype, order)
                return y
            else:
                return self.asarray0(y, dtype, order)
            
        self.asarray0 = np.asarray #save old one
        #numpy.asarray(a, dtype=None, order=None, *, like=None)
        #np.asarray = lambda y, **params: y.array if type(y) is Handler else self.asarray0(y, **params)
        #np.asarray = lambda y, **params: self.asarray0(y.array, **params) if type(y) is Handler else self.asarray0(y, **params)
        np.asarray = as_array_func

    def inject_handler_asanyarray(self):
        
        def as_anyarray_func(y, dtype=None, order=None, *, like=None):
            
            if type(y) is Handler:
                print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                y.array = self.asanyarray0(y.array, dtype, order)
                return y
            else:
                return self.asanyarray0(y, dtype, order)
                
        self.asanyarray0 = np.asanyarray #save old one
        #numpy.asanyarray(a, dtype=None, order=None, *, like=None)
        #np.asanyarray = lambda y, **params: y.array if type(y) is Handler else self.asanyarray0(y, **params)
        #np.asanyarray = lambda y, **params: self.asanyarray0(y.array, **params) if type(y) is Handler else self.asanyarray0(y, **params)
        np.asanyarray = as_anyarray_func
        
    def inject_handler_reshape(self):
        
        def reshape_func(y, newshape, order='C'):
            
            if type(y) is Handler:
                print("1rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                y.array = self.reshape0(y.array, newshape, order='C')
                return y
            else:
                print("2rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
                return self.reshape0(y, newshape, order='C')
                
        self.reshape0 = np.reshape #save old one reshape
        np.reshape = reshape_func #set new reshape

    def restore_np_as_array(self):
        if hasattr(self, 'asarray0'):
            np.asarray = self.asarray0
            
    def restore_np_as_anyarray(self):
        if hasattr(self, 'asanyarray0'):
            np.asanyarray = self.asanyarray0
            
    def restore_np_reshape(self):
        if hasattr(self, 'reshape0'):
            np.reshape = self.reshape0

    def supports_np_asarray(self):
        test = np.asarray(self)
        shape = np.shape(test)
        return not len(shape) == 0

    def supports_np_asanyarray(self):
        #test = np.asanyarray(self) ###############changed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        test = np.asanyarray(self.array)
        shape = np.shape(test)
        return not len(shape) == 0
    
    def supports_np_reshape(self):
        #test = np.asanyarray(self) ###############changed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # test = np.reshape(self.array)
        # shape = np.shape(test)
        # return not len(shape) == 0
        return False #not actual check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #add flatten????????????????????????????????????????????
    
    @property
    def shape(self):
        return self.array.shape

    def __str__(self) -> str:
        return str(self.array)
    
    def __getitem__(self, a):
        print("ccccccccccccccccccccccccccccccccc")
        return Handler(self.array.__getitem__(a)) 
        #return self.array.__getitem__(a) 

    def __setitem__(self, a, b):
        return self.array.__setitem__(a, b)
    
    def __del__(self): #restore functions
        print("=======================Handler destructor======================================")
        self.restore_np_as_array()
        self.restore_np_as_anyarray()
        self.restore_np_reshape()
        
     
    @property
    def dtype(self):
        return self.array.dtype
    
class P300Enchanced(P300):
    
    def get_data(self, dataset, subjects, return_epochs):   

        X, y, metadata = super().get_data(dataset, subjects, return_epochs)
        
        print("gggggggggggggggggggggggggggggggggggggggggggggggggggggggg")
        return X, Handler(y) , metadata

class DataAugment(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator='scm'):#FIX: one needs to select which class to augment
        """Init."""
        self.estimator = estimator 
        
    def fit(self, X, y=None):
        print("\nfit")
        return self
    
    def transform(self, X):
        print("\ntransform")
        return X
        
    def fit_transform(self, X, y):
        
        print("\nfit_transform")
        if type(y) is Handler:
            print("Y is Handler")
        else:
            print("Y is NOT Handler")
            sys.exit()
        
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
        # vae.summary()

        early_stop_loss = 'loss'
        #define two callbacks
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
        reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced

        print("Fit VAE")
        vae.fit(
            scaled_data, 
            batch_size = 32,
            epochs=5,
            shuffle = True,
            callbacks=[early_stop_callback, reduceLR],
            verbose = 1
        )
        
        #Final sampling from the vae
        num_samples = 100 #FIX: set to the correct number we need
        new_samples = vae.get_prior_samples(num_samples=num_samples)
        print("New samples generated")
        
        # inverse-transform scaling 
        new_samples = scaler.inverse_transform(new_samples)
        
        #y_new=np.empty(100); y_new.fill(1)
        
        #y.array = np.append(y.array, 1) #label sample 1
        #y.array = np.append(y.array, 1)
        
        return X


if __name__ == "__main__":
    
    # insert here your code
    pipelines = {}
    #XdawnCovariances can be used
    
    #should test with other than MDM pipelines!
    #pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result
    pipelines["DataAugment+MDM"] = make_pipeline(DataAugment(), Covariances("oas"), MDM(metric="riemann")) #requires xdawn to improve result
    
    print("Total pipelines to evaluate: ", len(pipelines))
    
    datasets = [BNCI2014008()]
    
    paradigm = P300Enchanced()
    
    evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=True)
    
    results = evaluation.process(pipelines)
    
    print(results.groupby('pipeline').mean('score')[['score', 'time']])