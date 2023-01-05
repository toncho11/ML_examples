# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:30:05 2023

@author: antona
"""

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM

from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
)

paradigm = P300()
le = LabelEncoder()

xdawn_filters_all = 4

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
    
    print("Building train data completed: ", X.shape)
    return X,y

class CovCNNClassifier(BaseEstimator, ClassifierMixin):

    def __buildModel(self, input_shape):
        
        #create model
        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
          
        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
          
        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
          
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
    
    def __init__(self, cov_mat_size, xdawn_filters = 4):
        
        # # Checking format of Image and setting the input shape
        if K.image_data_format() == 'channels_first': #these are image channels and here we do not have color, so it should be 1
            input_shape = (1, cov_mat_size, cov_mat_size)
        else:
            input_shape = (cov_mat_size, cov_mat_size, 1)
        print("Input shape:",input_shape)
        
        #self.covestm_train = None
        
        self.model = self.__buildModel(input_shape)
        
        self.xdawn_filters = xdawn_filters
    
    # def __buildCov(self, X, y):

    #     #Covariances, XdawnCovariances, ERPCovariances
    #     #covestm = Covariances(estimator="scm").fit()
    #     covestm = XdawnCovariances(nfilter = self.xdawn_filters, estimator="scm")#.fit(X,y)
    #     covmats = covestm.fit_transform(X,y)
        
    #     print("Covmats:", covmats.shape)
    #     return covmats, covestm
    
    def fit(self, X, y):
        #X, self.covestm_train = self.__buildCov(X, y)
        #print("Cov matrix size: ", X.shape)
        
        #should X_train be normalized between 0 and 1?
        callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X,  y, epochs = epochsTF, callbacks=[callback])
    
    def predict_proba(self, X):
        #X = self.covestm_train.transform(X)
        return self.model.predict(X)
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def predict(self, X):
        y_pred = self.predict_proba(X)
        y_pred = np.rint(y_pred)
        return y_pred
        
def EvaluateMDM(X_train, X_test, y_train, y_test):
    
    print ("Evaluating ...================================================================")
    
    #0 NonTarget
    #1 Target       
    print("Total train class target samples available: ", sum(y_train))
    print("Total train class non-target samples available: ", len(y_train) - sum(y_train))
    print("Total test class target samples available: ", sum(y_test))
    print("Total test class non-target samples available: ", len(y_test) - sum(y_test))
    
    clf = make_pipeline(XdawnCovariances(xdawn_filters_all), MDM())
    
    print("Training MDM...")
    clf.fit(X_train, y_train)
    
    print("Predicting MDM...")
    y_pred = clf.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy MDM #####: ", ba)
    print("Accuracy score    MDM #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     MDM #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s     P300: ", sum(y_pred), "/", sum(y_test))
    print("0s Non P300: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return ba

def EvaluateTF(X_train, X_test, y_train, y_test):
    
    print ("Evaluating ...================================================================")
    
    #0 NonTarget
    #1 Target       
    print("Total train class target samples available: ", sum(y_train))
    print("Total train class non-target samples available: ", len(y_train) - sum(y_train))
    print("Total test class target samples available: ", sum(y_test))
    print("Total test class non-target samples available: ", len(y_test) - sum(y_test))
    
    cov_mat_size = 16
    clf = make_pipeline(XdawnCovariances(xdawn_filters_all), CovCNNClassifier(cov_mat_size))
    
    print("Training TF...")
    clf.fit(X_train, y_train)
    
    print("Predicting TF...")
    y_pred = clf.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy TF #####: ", ba)
    print("Accuracy score    TF #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     TF #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s     P300: ", sum(y_pred), "/", sum(y_test))
    print("0s Non P300: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return ba

if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    #name, electrodes, subjects
    #bi2013a	    16	24
    #bi2014a    	16	64
    #BNCI2014009	16	10
    #BNCI2014008	 8	 8
    #BNCI2015003	 8	10
    #bi2015a        32  43
    ds = [BNCI2014009()] #16ch: BNCI2014009(), bi2014a(), bi2013a(); 8ch: BNCI2014008(), BNCI2015003(), 
    iterations = 10
    epochsTF = 50 #more means better training of CNN
    selectedSubjects = list(range(1,11))

    # init
    pure_mdm_scores = []
    tf_scores = []
    #aug_filtered_vs_all = [] #what portion of the newly generated samples looked like P300 according to MDM
    
    X, y = BuidlDataset(ds, selectedSubjects)

    for i in range(iterations):
        
        print("Iteration: ", i)
        
        #shuffle
        for x in range(20):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = np.array(X)[indices]
            y = np.array(y)[indices]
            
        #stratify - ensures that both the train and test sets have the proportion of examples in each class that is present in the provided “y” array
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) #, stratify = y
        #X_train, X_test, y_train, y_test = TrainSplitEqualBinary(X , y, 200)
        
        #shuffle
        for x in range(20):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = np.array(X_train)[indices]
            y_train = np.array(y_train)[indices]
            
        #MDM
        ba_mdm = EvaluateMDM(X_train, X_test, y_train, y_test)
        pure_mdm_scores.append(ba_mdm)
            
        #Tensorflow image classification
        ba_tf  = EvaluateTF(X_train, X_test, y_train, y_test)
        tf_scores.append(ba_tf)
        print('\nTF / MDM:', ba_tf ," / ", ba_mdm)    
        print("_______________________________________ end iteration")

print("===================================================")        
print("Mean of Test dataset balanced accuracy  TF:", np.mean(tf_scores))
print("Mean of Test dataset balanced accuracy MDM:", np.mean(pure_mdm_scores))