# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:56:26 2022

@author: antona

Trains a TimeVAE - a variational autoencoder on the P300 class in ERP EEG datasets. 
It tries to generate data for the P300 class - it performs a data augmentation.
Next it uses MDM from PyRiemann to classify the data (with and without data augmentation)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

#not valid: bi2012a, bi2013a
#bi2014a - 71 subjects
from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
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
#import sklearn.metrics
from sklearn.model_selection import train_test_split

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

#START CODE

paradigm = P300()

le = LabelEncoder()

# Puts all subjects in single X,y
def BuidlDataset(datasets):
    
    X = np.array([])
    y = np.array([])
    
    for dataset in datasets:
        
        subjects  = enumerate(dataset.subject_list)

        for subject_i, subject in subjects:
            
            #if subject_i > 2:
            #    break
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
            
            #X1 = np.transpose(X1)
            
            if (X.size == 0):
                X = np.copy(X1)
                y = np.copy(y1)
            else:
                X = np.concatenate((X, X1), axis=0)
                y = np.concatenate((y, y1), axis=0)
    
    print("Building train data completed: ", X.shape)
    return X,y

# should be changed to be K-fold
# http://moabb.neurotechx.com/docs/auto_tutorials/tutorial_3_benchmarking_multiple_pipelines.html
# PyRiemann MDM example: https://github.com/pyRiemann/pyRiemann/blob/master/examples/ERP/plot_classify_MEG_mdm.py
def Evaluate(X_train, X_test, y_train, y_test):
    
    print ("Evaluating ...================================================================")
    
    #0 NonTarget
    #1 Target       
    print("Total class target samples available: ", sum(y_train))
    print("Total class non-target samples available: ", len(y_train) - sum(y_train))

    n_components = 3 
    
    clf = make_pipeline(XdawnCovariances(n_components), MDM())
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))
    print("Precion score: ", sklearn.metrics.precision_score(y_test, y_pred))
    
    print("1s: ", sum(y_pred), "/", sum(y_test))
    print("0s: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return cr
    

# Augments the p300 class with TimeVAE
def AugmentData(X, y, selected_class, samples_required):
    
    print("In fit_transform")
    
    selected_class_indices = np.where(y == selected_class)
    
    X = X[selected_class_indices,:,:]

    X = X[-1,:,:,:] #remove the first exta dimension
    
    print("Count of P300 samples used by the VAE: ", X.shape)
    
    #FIX: not the correct format (3360, 8, 257) but should be (3360, 257, 8) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # N, T, D = X.shape #N = number of samples, T = time steps, D = feature dimensions
    # print(N, T, D)
    # X = X.reshape(N,D,T)
    X.transpose(0,2,1)
    N, T, D = X.shape
    #print(N, T, D)
    
    # min max scale the data    
    scaler = utils.MinMaxScaler()        
   
    scaled_data = scaler.fit_transform(X)
    
    latent_dim = 8
    
    #vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
    vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[1000,500], )
    
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
        epochs=3000, #default 500 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        shuffle = True,
        #callbacks=[early_stop_callback, reduceLR],
        verbose = 1
    )
    
    #Final sampling from the vae
    num_samples = samples_required #FIX: set to the correct number we need
    new_samples = vae.get_prior_samples(num_samples=num_samples)
    print("New samples generated: ", new_samples.shape[0])
    
    # inverse-transform scaling 
    new_samples = scaler.inverse_transform(new_samples)
    
    #new_samples = new_samples.reshape(num_samples, D, T) #convert back to D,T
    X.transpose(0,1,2) #convert back to D,T
    return new_samples
    
if __name__ == "__main__":
    
    #select dataset to be used
    #ds = [BNCI2014009()] #BNCI2014008()
    #warning datasets must have the same number of electrodes
    ds = [bi2014a()]
    
    X, y = BuidlDataset(ds)
    
    #shuffle
    for x in range(20):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = np.array(X)[indices]
        y = np.array(y)[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 400, stratify = y)
        
    print('Test with PyRiemann, NO data augmentation')
    #This produces a base result to compare with
    CR1 = Evaluate(X_train, X_test, y_train, y_test)
    
    #Perform data augmentation with TimeVAE
    
    P300Class = 1 #1 corresponds to P300 samples
    NonTargetCount = len(y_train) - sum(y_train)
    #samples_required = NonTargetCount - sum(y_train)
    samples_required = 600 #default 100
    X_augmented = AugmentData(X_train, y_train, P300Class, samples_required)
    
    #add to X_train and y_train
    X_train = np.concatenate((X_train, X_augmented), axis=0)
    y_train = np.concatenate((y_train, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
    
    #shuffle the real training data and the augmented data before testing again
    for x in range(6):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = np.array(X_train)[indices]
        y_train = np.array(y_train)[indices]
    
    print('Test with PyRiemann, WITH data augmentation')
    CR2 = Evaluate(X_train, X_test, y_train, y_test)
    
    print(CR1)
    print(CR2)