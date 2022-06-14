import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

import numpy as np

from sklearn.preprocessing import LabelEncoder

import Dither #pip install PyDither
import os
import glob
import time
import sys

from joblib import Parallel, delayed
from multiprocessing import Process

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

from mne.preprocessing import Xdawn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

#END IMPORTS

paradigm = P300()

le = LabelEncoder()

#https://www.researchgate.net/figure/Common-electrode-setup-for-P300-spellers-according-to-8-Eight-EEG-electrodes-are_fig1_221583051
#Common electrode setup for P300 spellers according to [8]. 
#Eight EEG electrodes are placed at Fz, Cz, P3, Pz, P4, PO7, Oz and PO8
#bi2013a: FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2

def multivariateRP(sample, electrodes, dimension, time_delay, percentage):
    
    channels_N = sample.shape[0]
    
    #Time window = T
    #delta = 40, the interval T is chpped into epochs of delta elements 
    #T is the time interval to be taken from the epoch sample beginning
       
    delta = time_delay 
    points_n = dimension
    
    #we need to leave enough space at the end to perform n=dimension jumps over time_delay data
    #otherwise the vectors will not be filled with the same amount of data
    T = sample.shape[1] - ((dimension-1) * time_delay)
    
    if ( T <= 0 ):
        print("Error in T multivariateRP ")
        sys.exit(1)
     
    print("T=",T, "/", sample.shape[1])
    X_traj = np.zeros((T,points_n * channels_N))
            
    for i in range(0,T): #delta is number of vectors with  length points_n
        
        for j in range(0,points_n):
            start_pos = j * delta
            pos = start_pos + i
            
            for e in electrodes:
                #print(e)
                pos_e = (e * points_n) + j
                #print(pos_e)
                #all points first channel, 
                X_traj[i, pos_e ] = sample[e,pos] #i is the vector, j is indexing isnide the vector 
            #print(pos)
            
    X_dist = np.zeros((T,T))
    
    #calculate distances
    for i in range(0,T): #i is the vector
        for j in range(0,T):
             v1 = X_traj[i,:]
             v2 = X_traj[j,:]
             X_dist[i,j] = np.sqrt( np.sum((v1 - v2) ** 2) ) 
    
    #out = Dither.dither(X_dist, 'floyd-steinberg', resize=True)
    out = X_dist
    
    return out


images_all = []
labels_all = []

def ProcessSamples(samples, X, y, folder, subject, m, tau , electrodes, percentage):

    global images_all
    global labels_all
    
    for sample_i in samples:
        print("Process Sample:",sample_i)
        label = y[sample_i]
        sample = X[sample_i]
 
        single_epoch_subject_rp = multivariateRP(sample, electrodes, m, tau, percentage)
    
        images_all.append(single_epoch_subject_rp)
        labels_all.append(label)
        # if ( label == 1 ):
        #     images_class_1.append(single_epoch_subject_rp)
        # elif ( label == 2 ):
        #     images_class_2.append(single_epoch_subject_rp)
        # else:
        #     print("Error")
            

def CreateData(dataset, m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject, enableXDAWN):
    
    subjects  = enumerate(dataset.subject_list[0:n_subjects])
    
    for subject_i, subject in subjects:
        
        epochs_class_1 = 0
        epochs_class_2 = 0
        
        print("Loading subject:" , subject) 
        
        #X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])

        #currently xDawn takes into account the entire dataset and not the reduced one that is used for the actual training
        if (enableXDAWN):
            X_epochs, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
            print("Performing XDawn")
            xd = Xdawn(n_components=6) #output channels = 2 * n_components
            X = np.asarray(xd.fit_transform(X_epochs))
            electrodes = range(X.shape[1])
            print("Finished XDawn")
        else:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        
        y = le.fit_transform(y)
        print(X.shape)  
        
        #check for errors
        if X.shape[2] < (m-1) * tau:
            print("Error: in m,tau and trial length")
            sys.exit(1)
        
        if (electrodes == []):
            electrodes = list(range(0,X.shape[1]))
        elif X.shape[1] < len(electrodes):
            print("Error: electrode list is longer than electrodes in dataset")
            sys.exit(1)  
        
        print("Electrodes selected: ",electrodes)
        #0 NonTarget
        #1 Target       
        print("Total class target samples available: ", sum(y))
        print("Total class non-target samples available: ", len(y) - sum(y))

        index_label1 = [];
        index_label2 = [];
        
        #get only the required number of samples
        for idx,val in enumerate(y):
            if (val == 0 and epochs_class_1 < max_epochs_per_subject):
                index_label1.append(idx)
                epochs_class_1 = epochs_class_1 + 1
            elif (val == 1 and epochs_class_2 < max_epochs_per_subject):
                index_label2.append(idx)
                epochs_class_2 = epochs_class_2 + 1
        
        
        #0 NonTarget
        #1 Target       
        print("Class target samples to be used: ", epochs_class_2)
        print("Class non-target samples to be used: ", epochs_class_1)
        
        print("Processing")
        n_jobs = 9
        processes = [None] * n_jobs            
        i=0          
        parallel = False
        
        if (parallel):
        
            print("Starting parallel processes")
            
            for section in np.array_split(index_label1 + index_label2 , n_jobs):
                processes[i] = Process(target=ProcessSamples,args=(section, X, y, "", subject, m, tau, electrodes, percentage))
                processes[i].start()
                print(i)
                i = i + 1
            
            print("Setting threads to join:")
            for p in processes:
                 p.join()
        else:
            ProcessSamples(index_label1 + index_label2, X, y, "", subject, m, tau, electrodes, percentage)
                         
def ProcessFolder(epochs_all_subjects, label_all_subjects):
    
    #build model
    
    #to prevent overfitting
    # - adjust the number of epochs
    # - smaller model
    # - regularizing L1, L2 or both
    # - dropout layers

    img_size1, img_size2 = np.array(epochs_all_subjects[0]).shape
    print(img_size1, img_size2)
    
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(img_size1,img_size2,1)))
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
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', #adam, rmsprop
                  metrics=['accuracy'])

    iterations = 1
    average_classification = 0;
        
    for i in range(iterations):
        
        print("Iteration: ",i)
        indices = np.arange(len(epochs_all_subjects))
        np.random.shuffle(indices)
        all_images_shuffled = np.array(epochs_all_subjects)[indices]
        labels_shuffled = np.array(label_all_subjects)[indices]
            
        #shuffle
        for s in range(0,20):
            print(s)
            indices = np.arange(len(all_images_shuffled))
            np.random.shuffle(indices)
            all_images_shuffled = np.array(all_images_shuffled)[indices]
            labels_shuffled = np.array(labels_shuffled)[indices]
        
        #split
        # X_train, X_test, y_train, y_test = train_test_split(all_images_shuffled, labels_shuffled, test_size=0.2)
        
        # X_train = np.array(X_train)[:, :, :, np.newaxis] #np.newaxis required by Keras
        # X_test = np.array(X_test)[:, :, :, np.newaxis]
        # y_train = np.array(y_train)
        # y_test = np.array(y_test)
        
        print("Class non-target samples: ", len(labels_shuffled) - sum(labels_shuffled))
        print("Class target samples: ", sum(labels_shuffled))

        epochs = 50;
        #model.fit(X_train, y_train, epochs=5, validation_data = (X_test,y_test) ) #validation_data=(X_test,y_test)
        #history = model.fit(np.array(epochs_all_subjects)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        #history = model.fit(np.array(all_images_shuffled)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        
        data_to_process = np.array(all_images_shuffled).astype('float32') #[:, :, :, np.newaxis]
        data_to_process = data_to_process.reshape(data_to_process.shape[0],data_to_process.shape[1],data_to_process.shape[2],1)
        print(data_to_process.shape)
        print("Start model fit")
        history = model.fit(data_to_process,  np.array(labels_shuffled), epochs=epochs, validation_split=0.2, shuffle=False )
        
        #sample = data_to_process[4]   
        #plt.imshow(sample, cmap = plt.cm.binary, origin='lower')
        
        #print("Validation accuracy = ", history.history['val_accuracy'][epochs-1])
        return history.history['val_accuracy'][epochs-1]

max_score = -1;
saved_params = []

def SearchHyperParameters(f1,f2):
    
    Subjects = 1; #default 10
    SamplesPerClass = 30; #default 1000
    
    #rp search
    #electrode search
    #CNN build 

    global images_all
    global labels_all
    

    for m in range(2,5):
        for tau in range (20,22):
            
            CreateData( BNCI2014008(), m, tau, f1, f2, [], Subjects , 5 , SamplesPerClass, False)
            score = ProcessFolder(images_all, labels_all)
            if (score > max_score):
                saved_params = [m,tau,score]
            images_all = []
            labesl_all = []
            
    print(saved_params)

if __name__ == '__main__':

    start = time.time()
    f1 = paradigm.filters[0][0]
    f2 = paradigm.filters[0][1]

    SearchHyperParameters(f1,f2)
    
    end = time.time()
    print("Elapsed time (in seconds):",end - start)
    
    
