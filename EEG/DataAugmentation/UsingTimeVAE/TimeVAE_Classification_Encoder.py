# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:56:26 2022

@author: antona

Trains a TimeVAE - a variational autoencoder on the P300 class in ERP EEG datasets. 
It uses TimeVAE to train an Encoder,
Next the Encoder is used for classification

Heplful: https://keras.io/examples/generative/vae/
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
import gc

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
from pyriemann.classification import KNearestNeighbor

#START CODE

paradigm = P300()

le = LabelEncoder()

def GetDataSetInfo(ds):
    #print("Parameters: ", ds.ds.__dict__)
    print("Dataset name: ", ds.__class__.__name__)
    print("Subjects: ", ds.subject_list)
    print("Subjects count: ", len(ds.subject_list))
    
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=[ds.subject_list[0]])
    
    print("Electrodes count (inferred): ", X.shape[1])
    print("Epoch length (inferred)    : ", X.shape[2])
    #print("Description:    : ", ds.__doc__)
    
# Puts all subjects in single X,y
def BuidlDataset(datasets, selectedSubjects):
    
    X = np.array([])
    y = np.array([])
    
    for dataset in datasets:
        
        GetDataSetInfo(dataset)
        
        dataset.subject_list = selectedSubjects
        
        subjects  = enumerate(dataset.subject_list)

        for subject_i, subject in subjects:
            
            # if subject_i > 0:
            #     break
            
            print("Loading subject:" , subject) 
            
            X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            
            y1 = le.fit_transform(y1)
            print(X1.shape)  
            
            #0 NonTarget
            #1 Target       
            print("Total class target samples available: ", sum(y1))
            print("Total class non-target samples available: ", len(y1) - sum(y1))
            
            # bi2014a: (102, 307)
            # BNCI2014009 (19,136)
            start = 19
            end = 136
            X1 = X1[:,:,start:end] #select just a portion of the signal around the P300
            
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
    print("Total train class target samples available: ", sum(y_train))
    print("Total train class non-target samples available: ", len(y_train) - sum(y_train))
    print("Total test class target samples available: ", sum(y_test))
    print("Total test class non-target samples available: ", len(y_test) - sum(y_test))

    n_components = 3 
    
    clf = make_pipeline(XdawnCovariances(n_components), MDM())
    #clf = make_pipeline(XdawnCovariances(n_components),  KNearestNeighbor(n_neighbors=1, n_jobs=10))
    
    print("Training...")
    clf.fit(X_train, y_train)
    
    print("Predicting...")
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
def TrainVAE(X, y, selected_class, iterations, hidden_layer_low, latent_dim, onlyP300):
    
    print("In TrainVAE")
    
    if onlyP300 == True:
    
        selected_class_indices = np.where(y == selected_class)
        
        X = X[selected_class_indices,:,:]
    
        X = X[-1,:,:,:] #remove the first exta dimension
    
    print("Count of P300 samples used by the VAE train: ", X.shape)
    
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
    
    early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced

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

def PlotEpoch(epoch):
    
    import pandas as pd
    import plotly.express as px
    
    epoch = epoch[-1,:,:]
    epoch = epoch.transpose(1,0)
    
    df = pd.DataFrame(data = epoch)
    
    for i in range(0, len(df.columns), 1):
        df.iloc[:, i]+= i * 30

    fig = px.line(df, width=600, height=400)
    
    fig.show(renderer='browser')
    
def PlotEpochs(epochs):
    
    import pandas as pd
    import plotly.express as px
    
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

def SaveVAEModel(model, model_dir, fname):
    model.save(model_dir, fname)

def LoadVAEModel(model_dir, fname): #problem ????????
    new_vae = TimeVAE.load(model_dir, fname)
    return new_vae

def CalculateMeanEpochs(epochs):   
    import tensorflow as tf    
    res = tf.math.reduce_mean(epochs, axis = 0, keepdims = True)
    return res.numpy()

# Returns a Test dataset that contains an equal amounts of each class
# y should contain only two classes 0 and 1
def TrainSplitEqualBinary(X, y, samples_n): #samples_n per class
    
    indicesClass1 = []
    indicesClass2 = []
    
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1 and len(indicesClass2) < samples_n:
            indicesClass2.append(i)
            
        if len(indicesClass1) == samples_n and len(indicesClass2) == samples_n:
            break
    
    X_test_class1 = X[indicesClass1]
    X_test_class2 = X[indicesClass2]
    
    X_test = np.concatenate((X_test_class1,X_test_class2), axis=0)
    
    #remove x_test from X
    X_train = np.delete(X, indicesClass1 + indicesClass2, axis=0)
    
    Y_test_class1 = y[indicesClass1]
    Y_test_class2 = y[indicesClass2]
    
    y_test = np.concatenate((Y_test_class1,Y_test_class2), axis=0)
    
    #remove y_test from y
    y_train = np.delete(y, indicesClass1 + indicesClass2, axis=0)
    
    if (X_test.shape[0] != 2 * samples_n or y_test.shape[0] != 2 * samples_n):
        raise Exception("Problem with split 1!")
        
    if (X_train.shape[0] + X_test.shape[0] != X.shape[0] or y_train.shape[0] + y_test.shape[0] != y.shape[0]):
        raise Exception("Problem with split 2!")
    
    return X_train, X_test, y_train, y_test

# generates interplated samples from the original samples
def GenerateInterpolated(X, y , selected_class):
    
    print ("Generating interpolated samples...")
    
    every_ith_sample = 10
    
    Xint = X.copy()
    epoch_length = X.shape[2]
    channels     = X.shape[1]
    indicesSelectedClass = []
    
    for i in range(0, len(y)):
        if y[i] == selected_class:
            for j in range(0,epoch_length,every_ith_sample):
                if j - 1 > 0 and j + 1 < epoch_length:
                    for m in range(0,channels):
                        #modify it
                        Xint[i,m,j] = (X[i,m,j-1] + X[i,m,j+1]) / 2
            indicesSelectedClass.append(i)
    
    print("Interpolated samples generated:", len(indicesSelectedClass))
    return Xint[indicesSelectedClass,:,:], np.repeat(selected_class,len(indicesSelectedClass))

def EvaluateWithEncoder(timeVAE, scaler, X_train, X_test, y_train, y_test):
    
    #There are 3 outputs of the VAE Encoder
    #encoder = [z_mean, z_log_var, encoder_output]
    #z_mean, z_log_var, z = self.encoder(data)
    output = 2
    
    #1) Use the auto encoder to produce feature vectors
    
    # Process X_train
    X_trained_c = X_train.copy()
    
    # convert to correct format expected by timeVAE
    N, T, D = X_trained_c.shape #N = number of samples, T = time steps, D = feature dimensions
    print(N, T, D)
    X_trained_c = X_trained_c.transpose(0,2,1)
    N, T, D = X_trained_c.shape
    print(N, T, D)
    
    X_train_scaled = scaler.fit_transform(X_trained_c)
    
    X_train_fv = timeVAE.encoder.predict(X_train_scaled)
    X_train_fv_np = np.array(X_train_fv)[ output ]
    
    # Process X_test
    X_test_c = X_test.copy()
    X_test_c = X_test_c.transpose(0,2,1)
    
    X_test_scaled = scaler.fit_transform(X_test_c)
    
    X_test_fv  = timeVAE.encoder.predict(X_test_scaled)
    X_test_fv_np = np.array(X_test_fv)[ output ]
   
    
    #2) Instantiate the Support Vector Classifier (SVC)
    from sklearn.svm import LinearSVC, SVC

    clf = SVC(C=1.0, random_state=1, kernel='rbf', verbose=False)
    #clf = LinearSVC(C=1.0, random_state=1, dual=False, verbose=False)
 
    # Fit the model
    print("Training standard classifier ...")
    clf.fit(X_train_fv_np, y_train)

    print("Predicting standard classifier ...")
    y_pred = clf.predict(X_test_fv_np)
    
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
    
if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    ds = [BNCI2014009()] #bi2014a() 
    iterations = 5
    iterationsVAE = 300 #more means better training
    selectedSubjects = list(range(1,3))
    
    # init
    pure_mdm_scores = []
    aug_mdm_scores = []
    
    X, y = BuidlDataset(ds, selectedSubjects)
        
    for i in range(iterations):
        
        #shuffle
        for x in range(20):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = np.array(X)[indices]
            y = np.array(y)[indices]
            
        #stratify - ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y” array
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) #, stratify = y
        #X_train, X_test, y_train, y_test = TrainSplitEqualBinary(X , y, 20)
        
        original_n_count = X_train.shape[0]
        
        #shuffle
        for x in range(20):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = np.array(X_train)[indices]
            y_train = np.array(y_train)[indices]
        
        #Perform data augmentation with TimeVAE
        
        P300Class = 1 #1 corresponds to P300 samples
        
        meanTrainOrig    = CalculateMeanEpochs( X_train[y_train == P300Class])
        meanTest         = CalculateMeanEpochs( X_test [y_test  == P300Class])
        meanNonP300Train = CalculateMeanEpochs( X_train[y_train == 0])
        meanNonP300Test  = CalculateMeanEpochs( X_test [y_test  == 0])
        
        P300ClassCount = sum(y_train)
        NonTargetCount = len(y_train) - P300ClassCount
        #samples_required = NonTargetCount - sum(y_train)
        #samples_required = 600 #default 100
        
        print('Test with PyRiemann, NO data augmentation')
        #This produces a base result to compare with
        CR1, pure_mdm_ba, modelMDM = Evaluate(X_train, X_test, y_train, y_test)
        pure_mdm_scores.append(pure_mdm_ba)
        #print(CR1)

        for hl in [700]:#700, 900, 2000 , default 500
            for ls in [12]: #16 produces NaNs
                print ("hidden layers low:", hl)
                print ("latent_dim:", ls)
                
                addInterpolated = True
                
                if addInterpolated:
                    #Addding samples by revmoving some data and replacing it with interpolated one
                    X_interpolated, y_interpolated = GenerateInterpolated(X_train, y_train, P300Class)
                    
                    if (P300ClassCount != X_interpolated.shape[0]):
                        raise Exception("Problem with interpolated data 1!")
                        
                    X_train = np.concatenate((X_train, X_interpolated), axis=0)
                    y_train = np.concatenate((y_train, y_interpolated), axis=0)
                
                for x in range(10):
                        indices = np.arange(len(X_train))
                        np.random.shuffle(indices)
                        X_train = np.array(X_train)[indices]
                        y_train = np.array(y_train)[indices]
                        
                #train and generate samples
                modelVAE, scalerVAE = TrainVAE(X_train, y_train, P300Class, iterationsVAE, hl, ls, False) #latent_dim = 8
                
                CR2, ba_augmented, _ = EvaluateWithEncoder(modelVAE, scalerVAE, X_train, X_test, y_train, y_test)
                                    
                aug_mdm_scores.append(ba_augmented)
                
                #print(CR2)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        del X_train, X_test, y_train, y_test
        gc.collect()

print("classification original / classification augmented:", np.mean(pure_mdm_scores), "/", np.mean(aug_mdm_scores), "Difference: ",np.mean(pure_mdm_scores) - np.mean(aug_mdm_scores))
