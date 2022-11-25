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
    print("Back to original dimensions: ", X.shape)
    
    return new_samples
    
def GenerateSamplesMDMfiltered(modelVAE, scaler, samples_required, modelMDM, selected_class):
    
    print("GenerateSamplesMDMfiltered start")
    good_samples_count = 0
    batch_samples_count = 10000
    X = np.array([])
    
    while (good_samples_count < samples_required):
        
        #Sampling from the VAE
        print("New samples requested: ", samples_required)
        new_samples = modelVAE.get_prior_samples(num_samples=batch_samples_count)
        print("Number of new samples generated: ", new_samples.shape[0])
        
        # inverse-transform scaling 
        new_samples = scaler.inverse_transform(new_samples)
        
        print("X augmented NANs: ", np.count_nonzero(np.isnan(new_samples)))
        
        print("Back to original dimensions: ", X.shape)
        new_samples = new_samples.transpose(0,2,1) #convert back to D,T
        
        print("classify")
        #classify
        y_pred = modelMDM.predict(new_samples)
        
        print("sum(y_pred):", sum(y_pred))
        
        #select only the P300 samples
        filtered_samples =  new_samples[y_pred == selected_class]
        
        if (sum(y_pred) != filtered_samples.shape[0]):
            print("WARNING: potential error")
        
        if (filtered_samples.shape[0] > 0):
            
            samples_still_needed = samples_required - good_samples_count
            
            samples_tobe_taken = min(samples_still_needed, filtered_samples.shape[0])
            
            if (X.size == 0):
                X = np.copy(filtered_samples[0:samples_tobe_taken,:,:])
            else:
                X = np.concatenate((X, filtered_samples[0:samples_tobe_taken,:,:]), axis=0)
                
            good_samples_count = good_samples_count + samples_tobe_taken
                
            print("New filtered samples added:", good_samples_count, "/", samples_required)
        else:
            print("Non added samples: ", new_samples.shape[0])
            print("Samples still needed: ", samples_required - good_samples_count, "/", samples_required)
        
    print("GenerateSamplesMDMfiltered end")
    return X

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

# returns a Test dataset that contains an equal amounts of each class
# y should contain only two classes 0  and 1
def TrainSplitEqualBinary(X, y, samples_n): #samplesd_n per class
    
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
    

if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    ds = [BNCI2014009()] #bi2014a() 
    iterations = 5
    iterationsVAE = 30 #more means better training
    selectedSubjects = list(range(1,11))
    
    # init
    pure_mdm_scores = []
    aug_mdm_scores = []
    #aug_filtered_vs_all = [] #what portion of the newly generated samples looked like P300 according to MDM
    
    X, y = BuidlDataset(ds, selectedSubjects)
        
    for i in range(iterations):
        
        #shuffle
        for x in range(10):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = np.array(X)[indices]
            y = np.array(y)[indices]
            
        #stratify - ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y” array
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) #, stratify = y
        
        X_train, X_test, y_train, y_test = TrainSplitEqualBinary(X , y, 200)
        
        #shuffle
        for x in range(10):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = np.array(X_train)[indices]
            y_train = np.array(y_train)[indices]
        
        #Perform data augmentation with TimeVAE
        
        P300Class = 1 #1 corresponds to P300 samples
        
        meanTrainOrig = CalculateMeanEpochs(X_train[y_train == P300Class])
        meanTest      = CalculateMeanEpochs( X_test[y_test  == P300Class])
        
        P300ClassCount = sum(y_train)
        NonTargetCount = len(y_train) - P300ClassCount
        #samples_required = NonTargetCount - sum(y_train)
        #samples_required = 600 #default 100
        
        print('Test with PyRiemann, NO data augmentation')
        #This produces a base result to compare with
        CR1, pure_mdm_ba, modelMDM = Evaluate(X_train, X_test, y_train, y_test)
        pure_mdm_scores.append(pure_mdm_ba)
        #print(CR1)
        
        max_ba = -1
        max_hl = -1
        max_ls = -1
        max_percentage = -1

        for hl in [700]:#700, 900, 2000 , default 500
            for ls in [8]: #16 produces NaNs
                print ("hidden layers low:", hl)
                print ("latent_dim:", ls)
                #train and generate samples
                modelVAE, scaler = TrainVAE(X_train, y_train, P300Class, iterationsVAE, hl, ls) #latent_dim = 8
                
                print("Reports for augmented data:") 
                for percentageP300Added in [10]:#[2, 3, 10, 15, 20]: #, 5, 10, 20
                    
                    print("% P300 Added:", percentageP300Added)
                    
                    samples_required = int(percentageP300Added * P300ClassCount / 100)  #5000 #default 100
                    
                    X_augmented = GenerateSamples(modelVAE, scaler, samples_required)
                    #X_augmented = GenerateSamplesMDMfiltered(modelVAE, scaler, samples_required, modelMDM, P300Class)
                    meanOnlyAugmented = CalculateMeanEpochs(X_augmented)
                    #PlotEpochs(X_augmented[0:12,:,:]) #let's have look at the augmented data
                    
                    #add to X_train and y_train
                    X_train = np.concatenate((X_train, X_augmented), axis=0)
                    y_train = np.concatenate((y_train, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
                    
                    #meanOrigAndAugmented = CalculateMeanEpochs(X_train)
                    meanManyAugmentedSamples = CalculateMeanEpochs(GenerateSamples(modelVAE, scaler, 2000))
                    
                    #shuffle the real training data and the augmented data before testing again
                    for x in range(6):
                        indices = np.arange(len(X_train))
                        np.random.shuffle(indices)
                        X_train = np.array(X_train)[indices]
                        y_train = np.array(y_train)[indices]
                    
                    print('Test with PyRiemann, WITH data augmentation. Metrics:')
                    CR2, ba_augmented, _ = Evaluate(X_train, X_test, y_train, y_test)
                    
                    aug_mdm_scores.append(ba_augmented)
                    
                    if ba_augmented > max_ba:
                        max_ba = ba_augmented
                        max_hl = hl
                        max_ls = ls
                        max_percentage = percentageP300Added
                    
                    #print(CR2)
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        del X_train, X_test, y_train, y_test
        gc.collect()

meanEpochs = np.array([meanTrainOrig[-1,:,:], meanOnlyAugmented[-1,:,:], meanManyAugmentedSamples[-1,:,:], meanTest[-1,:,:] ])             
#print("max all:", max_ba, max_hl, max_ls, max_percentage)
print("classification original / classification augmented:", np.mean(pure_mdm_scores), "/", np.mean(aug_mdm_scores))
legend = ["Mean P300 train data","Mean P300 Only Augmented used","Mean 2000 Augmented","Mean P300 test"]
PlotEpochs(meanEpochs)