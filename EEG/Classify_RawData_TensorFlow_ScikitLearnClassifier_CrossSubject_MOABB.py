# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:30:05 2023

@author: antona

Classificaion of P300 with Deep Learning (CNN) on raw data. 

Uses a scikitlearn classifier and thus allows scikitlearn pipelines to be used.
This version uses the CrossSubject evaluation provided by MOABB.

Notes:
- Scikit-leran 1.2 requires the attribute classes_ is provided 
- Video memory might not be enough if tf.config.experimental.set_memory_growth is not used
"""

import numpy as np
import sys
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Rescaling
from tensorflow import map_fn
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
)

paradigm = P300()
le = LabelEncoder()

# Puts all subjects in single X,y
def BuidlDataset(datasets, selectedSubjects):
    
    X = np.array([])
    y = np.array([])
    
    for dataset in datasets:
        
        #GetDataSetInfo(dataset)
        
        dataset.subject_list = selectedSubjects
        
        subjects  = enumerate(dataset.subject_list)

        for subject_i, subject in subjects:
            
            print("Loading subject:" , subject) 
            
            X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            
            X1 = X1.astype('float32') #possible data loss? float32 should be the correct one
            
            y1 = le.fit_transform(y1)
            print(X1.shape)  
            
            #0 NonTarget
            #1 Target       
            print("Total class target samples available: ", sum(y1))
            print("Total class non-target samples available: ", len(y1) - sum(y1))
            
            # bi2014a: (102, 307)
            # BNCI2014009 (19,136)
            #start = 19
            #end = 136
            #X1 = X1[:,:,start:end] #select just a portion of the signal around the P300
            
            if (X.size == 0):
                X = np.copy(X1)
                y = np.copy(y1)
            else:
                X = np.concatenate((X, X1), axis=0)
                y = np.concatenate((y, y1), axis=0)
    
    print("Building data completed: ", X.shape)
    print("Total class target samples available: ", sum(y))
    print("Total class non-target samples available: ", len(y) - sum(y))
    return X,y

class CovCNNClassifier(BaseEstimator, ClassifierMixin):

    classes_ = np.array([0, 1])
    
    def __buildModel(self, input_shape):
        
        #https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
        ks = 2
        ps = 2
        
        #create model
        model = Sequential()
        #model.add(Rescaling(1./255, input_shape=input_shape))
        model.add(Conv2D(filters=32, kernel_size=(ks, ks), input_shape=input_shape)) #, input_shape=input_shape
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(ps, ps)))
          
        model.add(Conv2D(filters=32, kernel_size=(ks, ks)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(ps,ps)))
          
        # model.add(Conv2D(filters=64, kernel_size=(ks, ks)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(ps, ps)))
          
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # compile model
        model.compile(loss='binary_crossentropy',
                      #optimizer='rmsprop',
                      optimizer='adam',
                      #optimizer='SGD',
                      metrics=['accuracy'],
                      )
        
        return model
    
    def __init__(self, epochs):
        self.epochs = epochs
    
    def fit(self, X, y):
        
        #print("fit")

        # # Checking format of Image and setting the input shape
        if K.image_data_format() == 'channels_first': #these are image channels and here we do not have color, so it should be 1
            input_shape = (1, X.shape[1], X.shape[2])
        else:
            input_shape = (X.shape[1], X.shape[2], 1) #nomally this is used
        print("Fit train X shape:", X.shape)
        
        self.model = self.__buildModel(input_shape)
        
        #should X_train be normalized between 0 and 1?
        #callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X,  y, epochs = self.epochs, verbose = 0) #callbacks=[callback]
    
    def predict_proba(self, X):
        #https://datascience.stackexchange.com/questions/72296/predict-proba-for-binary-classifier-in-tensorflow
        y_proba = self.model.predict(X)
        
        y_pred_int = np.rint(y_proba).astype(int)
        
        if (len(np.unique(y_pred_int)) != 2):
            print("Warning: Problem with y prediction proba. Only one class detected!", np.unique(y_pred_int))
        
        column0 = y_proba[:,0:1]  # Use 0:1 as a dummy slice to maintain a 2d array
        new_column = 1.0 - column0
        y_proba_exta_column = np.hstack((y_proba, new_column))

        return y_proba_exta_column

    def fit_predict(self, X, y):
        #print("fit_predict")
        self.fit(X, y)
        return self.predict(X)
    
    def predict(self, X):
        #print("predict")
        #y_pred = self.predict_proba(X)
        y_pred = self.model.predict(X)
        y_pred = np.rint(y_pred).astype(int)
        
        if (len(np.unique(y_pred)) != 2):
            print("Problem with y prediction. Only one class detected!", np.unique(y_pred))
            sys.exit()
            
        return y_pred
    
if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    #https://github.com/toncho11/ML_examples/wiki/EEG-datasets
    #name, electrodes, subjects
    #bi2013a	    16	24 (normal)
    #bi2014a    	16	64 (usually low performance)
    #BNCI2014009	16	10 (usually high performance)
    #BNCI2014008	 8	 8
    #BNCI2015003	 8	10
    #bi2015a        32  43
    #bi2015b        32  44
    #ds = [bi2014a(), bi2013a()] #both 16ch, 512 freq
    #ds = [bi2015a(), bi2015b()] #both 32ch, 512 freq
    n = 10
    ds = [BNCI2014009()] #Warning all datasets different from BNCI2014009 have too big epochs to be fit in the video memory
    epochs = 45 #default 60
    xdawn_filters_all = 4 #default 4
    
    # init
    pure_mdm_scores = []
    tf_scores = []
    
    #create pipelines
    pipelines = {}
    pipelines["MDM"] = make_pipeline(XdawnCovariances(xdawn_filters_all), MDM())
    pipelines["TF"] =  make_pipeline(CovCNNClassifier(epochs))
    
    #CrossSubjectEvaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=ds,
        overwrite=True
    )

    results = evaluation.process(pipelines)
    
    print("Averaging the session performance:")
    print(results.groupby('pipeline').mean('score')[['score', 'time']])
   
