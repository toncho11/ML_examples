# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:56:26 2022

@author: antona

Trains a TimeVAE - a variational autoencoder on the P300 class in ERP EEG datasets. 
It tries to generate data for the P300 class - it performs a data augmentation.
Next it uses MDM from PyRiemann to classify the data (with and without data augmentation)
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy #####: ", ba)
    print("Accuracy score    #####: ", sklearn.metrics.accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s     P300: ", sum(y_pred), "/", sum(y_test))
    print("0s Non P300: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return cr, ba, clf
    

# Augments the p300 class with TimeVAE
# latent space = encoded space
def TrainVAE(X, y, selected_class, iterations, hidden_layer_low, latent_dim):
    
    print("In TrainVAE")
    
    selected_class_indices = np.where(y == selected_class)
    
    X = X[selected_class_indices,:,:]

    X = X[-1,:,:,:] #remove the first exta dimension
    
    print("Count of P300 samples used by the VAE: ", X.shape)
    
    #FIX: not the correct format (3360, 8, 257) but should be (3360, 257, 8) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    N, T, D = X.shape #N = number of samples, T = time steps, D = feature dimensions
    print(N, T, D)
    # X = X.reshape(N,D,T)
    X = X.transpose(0,2,1)
    N, T, D = X.shape
    print(N, T, D)
    
    # min max scale the data    
    scaler = utils.MinMaxScaler()        
   
    scaled_data = scaler.fit_transform(X)
    
    #vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
    vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[hidden_layer_low * 2, hidden_layer_low], )
    
    vae.compile(optimizer=Adam())
    # vae.summary()

    early_stop_loss = 'loss'
    #define two callbacks
    #early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
    #reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced
    
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=50) 
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=50)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced

    print("Fit VAE")
    vae.fit(
        scaled_data, 
        batch_size = 32, #default 32
        epochs=iterations, #default 500
        shuffle = True,
        #callbacks=[early_stop_callback, reduceLR],
        verbose = 0
    )
    
    return vae, scaler
   

def GenerateSamples(model, scaler, samples_required):
    
    #Sampling from the VAE
    print("New samples requested: ", samples_required)
    new_samples = model.get_prior_samples(num_samples=samples_required)
    print("Number of new samples generated: ", new_samples.shape[0])
    
    # inverse-transform scaling 
    new_samples = scaler.inverse_transform(new_samples)
    
    print("X augmented NANs: ", np.count_nonzero(np.isnan(new_samples)))
    
    new_samples = new_samples.transpose(0,2,1) #convert back to D,T
    print("Back to original dimensions: ", new_samples.shape)
    
    return new_samples
    
def CreateDataset():
    
    X_train = np.array([])
    y_train = np.array([])
    
    X_test = np.array([])
    y_test = np.array([])
    
    X_train_a = np.array([])
    y_train_a = np.array([])
    
    # X_test_a = np.array([])
    # y_test_a = np.array([])
    
    #we seperate each subject into Train Xy and TestXy
    #we train the VAE on the TrainX (P300 class)
    #we save both the augmented Train X dataset
    #and Test Xy for the classfication later
    
    for dataset in ds:
        
        subjects  = enumerate(dataset.subject_list[30:39])

        for subject_i, subject in subjects:
            
            # if subject_i > 0:
            #     break
            
            print("Loading subject:" , subject) 
            
            X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            
            y1 = le.fit_transform(y1)
            
            #shuffle
            for x in range(7):
                indices = np.arange(X1.shape[0])
                np.random.shuffle(indices)
                X1 = np.array(X1)[indices]
                y1 = np.array(y1)[indices]
            
    
            #stratify - ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y” array
            X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.10) #, stratify = y
        
            #Perform data augmentation with TimeVAE
            
            P300Class = 1 #1 corresponds to P300 samples
    
            max_ba = -1
            max_hl = -1
            max_ls = -1
            max_percentage = -1
            
            iterationsVAE = 500 #more means better training
            
            hl = 500 #hidden layer
            ls = 8 #latent space
            
            # for hl in [500]:#700, 900, 2000 , default 500
            #     for ls in [8]: #16 produces NaNs
            print ("hidden layers low:", hl)
            print ("latent_dim:", ls)
            
            #train and generate samples
            
            #train on single subject
            modelVAE, scaler = TrainVAE(X_train1, y_train1, P300Class, iterationsVAE, hl, ls) #latent_dim = 8
                    
            P300ClassCount = sum(y_train1)
            NonTargetCount = len(y_train1) - P300ClassCount
            
            percentageP300Added = 5              
            print("% P300 Added:", percentageP300Added)
            
            samples_required = int(percentageP300Added * P300ClassCount / 100)  #5000 #default 100
            
            X_augmented = GenerateSamples(modelVAE, scaler, samples_required)
            #X_augmented = GenerateSamplesMDMfiltered(modelVAE, scaler, samples_required, modelMDM, P300Class)
            
            #add to X_train and y_train
            if (X_train.size == 0):
                #add subject original data
                X_train = np.copy(X_train1)
                y_train = np.copy(y_train1)
                #add subject augmented data
                # X_train = np.concatenate((X_train, X_augmented), axis=0)
                # y_train = np.concatenate((y_train, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
                
            else:
                #add subject original data
                X_train = np.concatenate((X_train, X_train1), axis=0)
                y_train = np.concatenate((y_train, y_train1), axis=0)
                #add subject augmented data
                # X_train = np.concatenate((X_train, X_augmented), axis=0)
                # y_train = np.concatenate((y_train, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
                
            
            #add to X_test and y_test
            if (X_test.size == 0):
                X_test = np.copy(X_test1)
                y_test = np.copy(y_test1)
            else:
                X_test = np.concatenate((X_test, X_test1), axis=0)
                y_test = np.concatenate((y_test, y_test1), axis=0)
            
            # ============================================================================
            #add to X_train and y_train
            if (X_train_a.size == 0):
                #add subject original data
                # X_train_a = np.copy(X_train1)
                # y_train_a = np.copy(y_train1)
                #add subject augmented data
                X_train_a = np.copy(X_augmented)
                y_train_a = np.copy(np.repeat(P300Class,X_augmented.shape[0]))
                
            else:
                #add subject original data
                # X_train_a = np.concatenate((X_train_a, X_train1), axis=0)
                # y_train_a = np.concatenate((y_train_a, y_train1), axis=0)
                #add subject augmented data
                X_train_a = np.concatenate((X_train_a, X_augmented), axis=0)
                y_train_a = np.concatenate((y_train_a, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
                
            
            # #add to X_test and y_test
            # if (X_test_a.size == 0):
            #     X_test_a = np.copy(X_test1)
            #     y_test_a = np.copy(y_test1)
            # else:
            #     X_test_a = np.concatenate((X_test_a, X_test1), axis=0)
            #     y_test_a = np.concatenate((y_test_a, y_test1), axis=0)
                    
                
    return X_train, y_train, X_test, y_test, X_train_a, y_train_a
    
if __name__ == "__main__":
    
    #select dataset to be used
    #ds = [BNCI2014009()] #BNCI2014008()
    #warning datasets must have the same number of electrodes
    ds = [bi2014a()]
    
    pure_mdm_scores = []
    aug_mdm_scores = []
    
    for i in range(10):
        
        print("Iteration: ", i)
        #1) Evaluate original dataset
        
        X_train, y_train, X_test, y_test, X_train_a, y_train_a = CreateDataset()
        
        if (np.count_nonzero(np.isnan(X_train_a)) > 0):
            print("NaNs detected ... skipping")
            del X_train, y_train, X_test, y_test, X_train_a, y_train_a
            import gc
            gc.collect()
            continue
        
        #shuffle the real training data and the augmented data before testing again
        for x in range(3):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train = np.array(X_train)[indices]
            y_train = np.array(y_train)[indices]
            
        print('Test with PyRiemann, NO data augmentation')
        #This produces a base result to compare with
        CR1, pure_mdm_ba, modelMDM = Evaluate(X_train, X_test, y_train, y_test)
        print(CR1)
        
        pure_mdm_scores.append(pure_mdm_ba)
        
        #2) train on the final Train and Test dataset (that uses augmented data) 
        
        #merge
        X_train = np.concatenate((X_train, X_train_a), axis=0)
        y_train = np.concatenate((y_train, y_train_a), axis=0)
        
        #shuffle the real training data and the augmented data before testing again
        for x in range(7):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train = np.array(X_train)[indices]
            y_train = np.array(y_train)[indices]
        
        print('Test with PyRiemann, WITH data augmentation. Metrics:')
        CR2, ba_augmented, _ = Evaluate(X_train, X_test, y_train, y_test)
        
        aug_mdm_scores.append(ba_augmented)
        
        del X_train, y_train, X_test, y_test, X_train_a, y_train_a
        import gc
        gc.collect()
        
        #print(CR2)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    
    # print("max all:", max_ba, max_hl, max_ls, max_percentage)
    # print("orginal / best:", pure_mdm_ba, "/", max_ba)
    
    #print("orginal / best:", pure_mdm_ba, "/", ba_best)
    print("orginal / augmented:", np.mean(pure_mdm_scores), "/", np.mean(aug_mdm_scores))