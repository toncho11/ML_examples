# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 00:06:10 2022

@author: antona
"""

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# fixes error: https://discuss.tensorflow.org/t/optimization-loop-failed-cancelled-operation-was-cancelled/1524/27
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

def TrainGAN(X, y, selected_class, iterations):
    
    print("In TrainGAN")
    
    selected_class_indices = np.where(y == selected_class)
    
    X_1_class = X[selected_class_indices,:,:]

    X_1_class = X_1_class[-1,:,:,:] #remove the first exta dimension
    
    print("Count of P300 samples used by the VAE train: ", X_1_class.shape)
    
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    from ydata_synthetic.synthesizers import ModelParameters

    #Define Model Hyperparameters
    # Specific to TimeGANs
    
    seq_len = X_1_class.shape[2] # Timesteps
    n_seq   = X_1_class.shape[1] # Features
    
    #data must be in the format (epoch, samples, features)
    X_1_class = X_1_class.transpose(0,2,1)

    # Hidden units for generator (GRU & LSTM).
    # Also decides output_units for generator
    hidden_dim = 24

    gamma = 1           # Used for discriminator loss

    noise_dim = 32      # Used by generator as a starter dimension
    dim = 128           # UNUSED
    batch_size = 128

    learning_rate = 5e-4
    beta_1 = 0          # UNUSED
    beta_2 = 1          # UNUSED
    data_dim = 28       # UNUSED

    # batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
    gan_args = ModelParameters(batch_size=batch_size,
                               lr=learning_rate,
                               noise_dim=noise_dim,
                               layers_dim=dim)

    # Read the Input data
    from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

    #Training the TimeGAN synthetizer
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(X_1_class, train_steps=iterations)
    synth.save('synth_energy.pkl') #save trained model
        
    return synth
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
    
    print("Training MDM...")
    clf.fit(X_train, y_train)
    
    print("Predicting MDM...")
    y_pred = clf.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy MDM #####: ", ba)
    print("Accuracy score    MDM #####: ", sklearn.metrics.accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     MDM #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s     P300: ", sum(y_pred), "/", sum(y_test))
    print("0s Non P300: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return cr, ba, clf
    
#uses the VAE encoder to generate new data
def GenerateAugmentedSamples(model, samples_required):
    
    #Sampling from the VAE
    print("New samples requested: ", samples_required)
    
    new_samples = model.sample(samples_required)
    
    #switching from (epochs, timesteps, channels) to (epochs, channels, timesteps)
    new_samples = new_samples.transpose(0,2,1)
    
    print("Number of new samples generated: ", new_samples.shape[0])
    
    return new_samples
    
# def GenerateAugmentedSamplesMDMfiltered(modelVAE, scaler, samples_required, modelMDM, selected_class):
    
#     print("GenerateAugmentedSamplesMDMfiltered start")
#     good_samples_count = 0
#     batch_samples_count = 5000
#     Xf = np.array([]) #all new generated samples filtered by MDM
    
#     while (good_samples_count < samples_required):
        
#         #Sampling from the VAE
#         print("New samples requested: ", samples_required)
#         new_samples = modelVAE.get_prior_samples(num_samples=batch_samples_count)
#         print("Number of new samples generated: ", new_samples.shape[0])
        
#         # inverse-transform scaling 
#         new_samples = scaler.inverse_transform(new_samples)
        
#         print("X augmented NANs: ", np.count_nonzero(np.isnan(new_samples)))
        
#         print("Back to original dimensions: ", X.shape)
#         new_samples = new_samples.transpose(0,2,1) #convert back to D,T
        
#         print("classify")
#         #classify
#         y_pred = modelMDM.predict(new_samples)
        
#         print("sum(y_pred):", sum(y_pred))
        
#         #select only the P300 samples
#         new_filtered_samples =  new_samples[y_pred == selected_class]
        
#         new_filtered_samplesCount = new_filtered_samples.shape[0]
        
#         if (sum(y_pred) != new_filtered_samplesCount):
#             print("WARNING: potential error")
        
#         if (new_filtered_samplesCount > 0):
            
#             samples_still_needed = samples_required - good_samples_count
            
#             samples_tobe_taken = min(samples_still_needed, new_filtered_samplesCount)
            
#             if (Xf.size == 0):
#                 Xf = np.copy(new_filtered_samples[0:samples_tobe_taken,:,:])
#             else:
#                 Xf = np.concatenate((Xf, new_filtered_samples[0:samples_tobe_taken,:,:]), axis=0)
                
#             good_samples_count = good_samples_count + samples_tobe_taken
                
#             print("Total filtered samples added:", good_samples_count, "/", samples_required)
#         else:
#             print("Non added samples: ", new_samples.shape[0])
#             print("Samples still needed: ", samples_required - good_samples_count, "/", samples_required)
        
#     print("GenerateSamplesMGenerateAugmentedSamplesMDMfilteredDMfiltered end")
#     return Xf

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
def GenerateInterpolated(X,y, selected_class):
    
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

if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    ds = [BNCI2014009()] #bi2014a() 
    iterations = 5
    iterationsGAN = 50 #more means better training
    selectedSubjects = list(range(1,2))
    
    # init
    pure_mdm_scores = []
    aug_mdm_scores = []
    #aug_filtered_vs_all = [] #what portion of the newly generated samples looked like P300 according to MDM
    
    X, y = BuidlDataset(ds, selectedSubjects)
        
    for i in range(iterations):
        
        #shuffle
        for x in range(20):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = np.array(X)[indices]
            y = np.array(y)[indices]
            
        #stratify - ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y” array
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) #, stratify = y
        
        X_train, X_test, y_train, y_test = TrainSplitEqualBinary(X , y, 200)
        
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

        #train and generate samples
        modelGAN = TrainGAN(X_train, y_train, P300Class, iterationsGAN) #latent_dim = 8
        
        print("Reports for augmented data:") 
        for percentageP300Added in [100]:#[2, 3, 10, 15, 20]: #, 5, 10, 20
            
            print("% P300 Added:", percentageP300Added)
            
            samples_required = int(percentageP300Added * P300ClassCount / 100)  #5000 #default 100
            
            #1) Data Augmentation with GAN
            
            X_augmented = GenerateAugmentedSamples(modelGAN, samples_required)
            #X_augmented = GenerateAugmentedSamplesMDMfiltered(modelGAN, scaler, samples_required, modelMDM, P300Class)
            meanOnlyAugmented = CalculateMeanEpochs(X_augmented)
            #PlotEpochs(X_augmented[0:12,:,:]) #let's have look at the augmented data
            
            # #2) Addding samples by revmoving some data and replacing it with interplated data
            # X_interpolated, y_interpolated = GenerateInterpolated(X_train, y_train, P300Class)
            # if (P300ClassCount != X_interpolated.shape[0]):
            #     raise Exception("Problem with interpolated data 1!")
                
            # Adding both the augmented and interpolated data
            
            #1)
            
            X_train_old_count = X_train.shape[0]
            
            if (original_n_count != X_train_old_count):
                raise Exception("Problem with original data")
            
            #add the GAN augmented data to X_train and y_train 
            X_train = np.concatenate((X_train, X_augmented), axis=0)
            y_train = np.concatenate((y_train, np.repeat(P300Class,X_augmented.shape[0])), axis=0)
            
            if (X_train.shape[0] != X_augmented.shape[0] + X_train_old_count):
                raise Exception("Problem adding augmented data")
            
            #2) add interpolated data (only P300 class)
            # X_train = np.concatenate((X_train, X_interpolated), axis=0)
            # y_train = np.concatenate((y_train, y_interpolated), axis=0)
            
            #meanOrigAndAugmented = CalculateMeanEpochs(X_train)
            meanManyAugmentedSamples = CalculateMeanEpochs(GenerateAugmentedSamples(modelGAN, 2000))
            
            #shuffle the real training data and the augmented data before testing again
            for x in range(20):
                indices = np.arange(len(X_train))
                np.random.shuffle(indices)
                X_train = np.array(X_train)[indices]
                y_train = np.array(y_train)[indices]
            
            print('Test with PyRiemann, WITH data augmentation. Metrics:')
            CR2, ba_augmented, _ = Evaluate(X_train, X_test, y_train, y_test)
            
            aug_mdm_scores.append(ba_augmented)
            
            #print(CR2)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        del X_train, X_test, y_train, y_test, X_augmented
        gc.collect()

meanEpochs = np.array([meanTrainOrig[-1,:,:], meanOnlyAugmented[-1,:,:], meanManyAugmentedSamples[-1,:,:], meanTest[-1,:,:], meanNonP300Train[-1,:,:], meanNonP300Test[-1,:,:]])             
#print("max all:", max_ba, max_hl, max_ls, max_percentage)
print("classification original / classification augmented:", np.mean(pure_mdm_scores), "/", np.mean(aug_mdm_scores), "Difference: ",np.mean(pure_mdm_scores) - np.mean(aug_mdm_scores))
legend = ["Mean P300 train data","Mean P300 Only Augmented used","Mean 2000 Augmented","Mean P300 test", "Mean Non P300 Train", "Mean Non P300 test"]
PlotEpochs(meanEpochs)

