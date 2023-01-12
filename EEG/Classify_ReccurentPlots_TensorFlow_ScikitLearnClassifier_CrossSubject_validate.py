# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:30:05 2023

@author: antona

Generating reccurent plots and then classifying them using CNN and TensorFlow.

Uses a scikitlearn classifier and thus allows scikitlearn pipelines to be used.
"""

import numpy as np
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from skimage import data, io, filters

from moabb.datasets import bi2013a, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

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

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM

import Dither
#from mne.preprocessing import Xdawn
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

class ReccurentPlots(BaseEstimator, TransformerMixin):
    
    #https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    def __noisy(self, noise_typ, image):
       if noise_typ == "gauss":
          row,col= image.shape
          ch = 1 
          mean = 0
          var = 0.1
          sigma = var**0.5
          gauss = np.random.normal(mean,sigma,(row,col,ch))
          gauss = gauss.reshape(row,col,ch)
          noisy = image + gauss[-1]
          
          return noisy
  
    #@tf.function
    def __multivariateRP(self, sample):
        
        #print(sample.shape)
        
        electrodes = self.electrodes
        dimension  = self.dimension
        time_delay = self.time_delay
        percentage = self.percentage
        
        channels_N = sample.shape[0]
        
        #Time window = T
        #delta = 40, the interval T is chpped into epochs of delta elements 
        #T is the time interval to be taken from the epoch sample beginning
           
        delta = time_delay 
        points_n = dimension
        
        #we need to leave enough space at the end to perform n=dimension jumps over time_delay data
        #otherwise the vectors will not be filled with the same amount of data
        T = sample.shape[1] - ((dimension-1) * time_delay)
         
        #print("T=",T, "/", sample.shape[1])
        X_traj = np.zeros((T,points_n * channels_N),dtype='float32')
                
        for i in range(0,T): #delta is number of vectors with  length points_n
            
            for j in range(0,points_n):
                start_pos = j * delta
                pos = start_pos + i
                
                for e in electrodes:
                    #print(e)
                    pos_e = (int(e) * points_n) + j
                    #print(pos_e)
                    #all points first channel, 
                    X_traj[i, pos_e ] = sample[e, pos] #i is the vector, j is indexing isnide the vector 
                #print(pos)
                
        X_dist = np.zeros((T,T),dtype='float32')
        
        #calculate distances
        for i in range(0,T): #i is the vector
            for j in range(0,T):
                 v1 = X_traj[i,:]
                 v2 = X_traj[j,:]
                 X_dist[i,j] = np.sqrt( np.sum((v1 - v2) ** 2) ) 
        
        new_image = X_dist.copy()
        new_image[new_image > np.percentile(new_image,percentage)] = 0
        new_image_noise = self.__noisy("gauss", new_image)
        new_image_scaled = new_image_noise / 255.0
        return new_image_scaled
    
    def __init__(self, electrodes, dimension, time_delay, percentage):
        self.percentage = percentage
        self.electrodes = electrodes
        self.dimension  = dimension
        self.time_delay = time_delay
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
       if (len(self.electrodes) != X.shape[1]):
           print("Problem with number of electrodes and input data")
           sys.exit()
           
       print("ReccurentPlots transform start", X.shape)
       
       #return map_fn(self.__multivariateRP, X, parallel_iterations=10) #too slow and parallel processing does not work
       
       X = X.astype('float32')
           
       out = np.array([],dtype='float32')
       
       n = X.shape[0]
       
       for i in range(n):
           
            rp = self.__multivariateRP(X[i])
           
            resizeImage = False #resize image to fit into memory
            #tf.image.resize
            
            rp = rp[np.newaxis,...]
            if (out.size == 0):
                out = np.copy(rp)
            else:
                out = np.concatenate((out, rp), axis=0)
                
            if i % 200 == 0:
                print(i, " / ", n)
               
       print("ReccurentPlots transform end", out.shape)
       return out


class CovCNNClassifier(BaseEstimator, ClassifierMixin):

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
          
        model.add(Conv2D(filters=64, kernel_size=(ks, ks)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(ps, ps)))
          
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
        
        cov_mat_size = X.shape[1]
        
        # # Checking format of Image and setting the input shape
        if K.image_data_format() == 'channels_first': #these are image channels and here we do not have color, so it should be 1
            input_shape = (1, cov_mat_size, cov_mat_size)
        else:
            input_shape = (cov_mat_size, cov_mat_size, 1)
        print("Input shape:",input_shape)
        
        self.model = self.__buildModel(input_shape)
        
        #should X_train be normalized between 0 and 1?
        #callback = callbacks.EarlyStopping(monitor='loss', patience=3)
        self.model.fit(X,  y, epochs = self.epochs, verbose = 0) #callbacks=[callback]
    
    # def predict_proba(self, X):
    #     #print("predict_proba")
    #     #X = self.covestm_train.transform(X)
    #     return self.model.predict(X)

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
        
def EvaluateMDM(X_train, X_test, y_train, y_test):
    
    print ("Evaluating MDM start ...================================================================")
    
    #0 NonTarget
    #1 Target       
    print("Total train class target samples available: ", sum(y_train))
    print("Total train class non-target samples available: ", len(y_train) - sum(y_train))
    print("Total test class target samples available: ", sum(y_test))
    print("Total test class non-target samples available: ", len(y_test) - sum(y_test))
    
    clf = make_pipeline(XdawnCovariances(xdawn_filters_all), MDM())
    #clf = make_pipeline(Covariances(), MDM())
    
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
    
    #from sklearn.metrics import classification_report
    #cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return ba

#data must be very well shuffled before providing it to this method
def EvaluateTF(X_train, X_test, y_train, y_test, epochs):
    
    print ("Evaluating TF start ...================================================================")
    #0 NonTarget
    #1 Target       
    print("Total train class target samples available: ", sum(y_train))
    print("Total train class non-target samples available: ", len(y_train) - sum(y_train))
    print("Total test class target samples available: ", sum(y_test))
    print("Total test class non-target samples available: ", len(y_test) - sum(y_test))
    
    #get same data to evaluate and then choose the best model 
    
    validate = False
    
    if validate:
        p = 0.10 #percentage to be used as validation dataset
        n = X_train.shape[0]
        validate_count = int(np.rint(p * n))
        
        X_validate = X_train[0:validate_count,:,:]
        y_validate = y_train[0:validate_count]
        
        X_train = X_train[validate_count:n,:,:]
        y_train = y_train[validate_count:n]
    
    #electrodes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #must be improved!
    
    if validate:
        
        classifiers_count = 5
        models = []
        results_ba = []

        electrodes = list(range(0, X_train.shape[1])) 
        for x in range(classifiers_count):
            
            
            #clf = make_pipeline(XdawnCovariances(xdawn_filters_all), CovCNNClassifier(epochs))
            clf = make_pipeline(ReccurentPlots(electrodes, 6, 30, 20), CovCNNClassifier(epochs))
            models.append(clf)
            
            if (len(np.unique(y_train)) != 2):
                print("Problem with y_train.")
                sys.exit()
            
            print("Fitting TF",x,"...")
            
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_validate)
            
            if (len(np.unique(y_pred)) != 2):
                print("Warning y_pred has only one class!")
                
            if (len(np.unique(y_validate)) != 2):
                print("Problem with y validate.")
                sys.exit()
                
            print("Calculating balanced accuracy TF",x,"...")
            ba_current = balanced_accuracy_score(y_validate, y_pred)
            results_ba.append(ba_current)
        
        print("Results TF models:", results_ba)
        max_ind = np.argmax(results_ba)
        print("Selected TF model:", max_ind)
        
        y_pred = models[max_ind].predict(X_test)
        ba = balanced_accuracy_score(y_test, y_pred)
        
        return ba
    else:
        
        if (len(np.unique(y_train)) != 2):
            print("Problem with y_train.")
            sys.exit()
            
        #xd = Xdawn(n_components=4) #output channels = 2 * n_components
        #X_train = np.asarray(xd.fit_transform(X_epochs))
        #X_test = np.asarray(xd.fit_transform(X_epochs))
        electrodes = list(range(0, xdawn_filters_all * 2))
        electrodes = list(range(0, X_train.shape[1]))
        
        #clf = make_pipeline(Xdawn(nfilter=xdawn_filters_all), ReccurentPlots(electrodes, 6, 30, 20), CovCNNClassifier(epochs))
        clf = make_pipeline(ReccurentPlots(electrodes, 6, 30, 20), CovCNNClassifier(epochs))
        
        clf.fit(X_train, y_train)
        
        print("Predicting on TRAIN data ...")
        y_pred_train = clf.predict(X_train)
        ba_t = balanced_accuracy_score(y_train, y_pred_train)
        print("Balanced accuracy on TRAIN data TF", ba_t)
        
        if (len(np.unique(y_test)) != 2):
            print("Problem with y_test.")
            sys.exit()
        
        print("Predicting on TEST data ...")
        y_pred_test = clf.predict(X_test)
        
        ba = balanced_accuracy_score(y_test, y_pred_test)
        print("Balanced accuracy on TEST data TF",ba)
        return ba

def AdjustSamplesCount(X, y): #samples_n per class
    
    samples_n = sum(y)

    indicesClass1 = []
    indicesClass2 = []
    
    for i in range(0, len(y)):
        if y[i] == 0 and len(indicesClass1) < samples_n:
            indicesClass1.append(i)
        elif y[i] == 1: # and len(indicesClass2) < samples_n:
            indicesClass2.append(i)
    
    X_class1 = X[indicesClass1]
    X_class2 = X[indicesClass2]
    
    y = y[indicesClass1 + indicesClass2] 
    
    X = np.concatenate((X_class1,X_class2), axis=0)
    
    #shuffle because this function orders them
    for x in range(20):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = np.array(X)[indices]
        y = np.array(y)[indices]
        
    if (X.shape[0] != (len(indicesClass1) + len(indicesClass2))):
        print("Error AdjustSamplesCount")
        sys.exit()
    
    return X,y
    
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
    ds = [BNCI2014009()]
    epochs = 60 #default 60
    xdawn_filters_all = 4 #default 4
    train_data_adjustment_equal = False
    shuffle_train_data = True #required if train_data_adjustment_equal = true 
    
    # init
    pure_mdm_scores = []
    tf_scores = []
    
    for i in range(0,n):
        
        print("Iteration: ", i)
        
        all_subjects = list(range(1,n+1))
        
        current_test_subject = [all_subjects[i]]
        
        all_subjects.pop(i)
        
        subjects_train = all_subjects
        
        X_train, y_train = BuidlDataset(ds, subjects_train)
        X_test,  y_test  = BuidlDataset(ds, current_test_subject)
        
        #shuffle train before reducing the non P300 samples
        if (shuffle_train_data):
            for x in range(20):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train = np.array(X_train)[indices]
                y_train = np.array(y_train)[indices]
        
        
        #so maybe trainig only on the non P300 class works good 
        if (train_data_adjustment_equal):
            X_train,y_train = AdjustSamplesCount(X_train, y_train) #preserve class 1, limit class 0
            X_test,y_test   = AdjustSamplesCount(X_test,  y_test) #preserve class 1, limit class 0
        
        print("y check: ",np.unique(y_train), np.unique(y_test))
        if not (np.array_equal(np.unique(y_train),np.unique(y_test))):   
            print("Problem with y train/test!")
            sys.exit()
            
        #MDM
        #ba_mdm = EvaluateMDM(X_train, X_test, y_train, y_test)
        #pure_mdm_scores.append(ba_mdm)
            
        #Tensorflow image classification
        ba_tf  = EvaluateTF(X_train, X_test, y_train, y_test, epochs)
        tf_scores.append(ba_tf)
        
        #result
        #print('\n[[ TF / MDM:', ba_tf ," / ", ba_mdm," ]]")    
        print("_______________________________________ end iteration")
        del X_train,y_train,X_test,y_test

print("===================================================")
print(tf_scores)
print(pure_mdm_scores)         
print("Mean of Test dataset balanced accuracy  TF:", np.mean(tf_scores), "std:",np.std(tf_scores))
#print("Mean of Test dataset balanced accuracy MDM:", np.mean(pure_mdm_scores), "std:",np.std(pure_mdm_scores))
#print("Average difference:", round(np.mean(np.array(pure_mdm_scores)-np.array(tf_scores)), 2) * 100 , "%")