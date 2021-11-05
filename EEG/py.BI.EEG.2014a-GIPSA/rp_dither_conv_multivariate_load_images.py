import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

import mne
from pyts.image import RecurrencePlot
import gc
import os
from sklearn import svm
from sklearn.model_selection import train_test_split

"""
=============================
Classification of EGG signal from two states: eyes open and eyes closed.
Each sample of the two states is represented as an image (a recurrence plot),
next the images are classified using Deep Learning model. It uses rp images that
were previously generated
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2
Pyts 0.11 (a Python Package for Time Series Classification,exists in Anaconda, provides recurrence plots)

"""
# Authors: Anton Andreev
#
# License: BSD (3-clause)

'''
https://towardsdatascience.com/dont-overfit-how-to-prevent-overfitting-in-your-deep-learning-models-63274e552323
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
'''

import warnings
#warnings.filterwarnings("ignore")


#Parameters
#10-20 international system
#'Fp1','Fp2','Fc5','Fz','Fc6','T7','Cz','T8','P7','P3','Pz','P4','P8','O1','Oz','O2','stim'
#alpha is at the back of the brain
#start form 0
#electrode = 14 #get the Oz:14
#electrode = 5 #get the T7:5
#m = 5
#tau = 30 
m = 5 
tau = 30
#rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
#rp = RecurrencePlot(threshold=0.2, dimension = m, time_delay = tau, percentage=20)
rp = RecurrencePlot(threshold='point', dimension = m, time_delay = tau, percentage=20)
n_train_subjects = 20 #max=19
length_s = 20 #max=19
filter_fmin = 4 #default 3
filter_fmax = 13 #default 40
electrodes = [9,10,11,13,14,15]
#electrodes = [6,8,12,9,10,11,13,14,15]
#electrodes = list(range(0,16))
#folder = "D:\Work\ML_examples\EEG\py.ALPHA.EEG.2017-GIPSA\multivariate_rp_images"
   
def ProcessFolder(folder, n_max_subjects):

    epochs_all_subjects = [];
    label_all_subjects = [];
    
    print("Loading data:")
    
    images_loaded = 0
    for filename in os.listdir(folder):
        if filename.endswith(".npy"): 
            
            #print(os.path.join(folder, filename))
            base_name = os.path.basename(filename)
            
            parts = base_name.split("_")
            #print(parts)
            label = int(parts[4].split(".")[0])
            subject = int(parts[1])
            #print("Subject: ", subject, " Label: ", label)
            
            if (subject < n_max_subjects):
                images_loaded = images_loaded + 1
                rp_image=np.load(os.path.join(folder, filename))
                
                epochs_all_subjects.append(rp_image)
                
                label_all_subjects.append(label)
            
        else:
            continue
    
    print("Images loaded: ", images_loaded)
    
    #build model

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
         
        #shuffle
        # indices = np.arange(len(epochs_all_subjects))
        # np.random.shuffle(indices)
        # all_images_shuffled = np.array(epochs_all_subjects)[indices]
        # labels_shuffled = np.array(label_all_subjects)[indices]
        
        #split
        # X_train, X_test, y_train, y_test = train_test_split(all_images_shuffled, labels_shuffled, test_size=0.2)
        
        # X_train = np.array(X_train)[:, :, :, np.newaxis] #np.newaxis required by Keras
        # X_test = np.array(X_test)[:, :, :, np.newaxis]
        # y_train = np.array(y_train)
        # y_test = np.array(y_test)
        

        epochs = 30;
        #model.fit(X_train, y_train, epochs=5, validation_data = (X_test,y_test) ) #validation_data=(X_test,y_test)
        #history = model.fit(np.array(epochs_all_subjects)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        #history = model.fit(np.array(all_images_shuffled)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        
        data_to_process = np.array(epochs_all_subjects).astype('float32') #[:, :, :, np.newaxis]
        data_to_process = data_to_process.reshape(data_to_process.shape[0],data_to_process.shape[1],data_to_process.shape[2],1)
        print(data_to_process.shape)
        history = model.fit(data_to_process,  np.array(label_all_subjects), epochs=epochs, validation_split=0.2, shuffle=True )
        
        #sample = data_to_process[4]   
        #plt.imshow(sample, cmap = plt.cm.binary, origin='lower')
        
        print("Validation accuracy = ", history.history['val_accuracy'][epochs-1])
        return history.history['val_accuracy'][epochs-1]

        
#print("Test data:================================================================================================================")

data_folder="D:\\Work\\ML_examples\\EEG\\py.BI.EEG.2014a-GIPSA\\data"

results = []
max_folder = 20;
i = 0;

# for x in os.walk(data_folder):
#     target_folder = x[0]
#     if target_folder != data_folder and "rp_dither_" in target_folder and i < max_folder:
#         print("target_folder =",target_folder)
#         score = ProcessFolder(target_folder, 100)
#         print("======================================================================================")
#         r = [target_folder,score]
#         results.append(r)
#         i = i + 1
        
#ProcessFolder(data_folder + "\\rp_dither_m_5_tau_40_f1_1_f2_20_el_4_nsub_3_per_-1_nepo_20",100)
ProcessFolder(data_folder + "\\rp_dither_m_5_tau_40_f1_1_f2_20_el_4_nsub_5_per_-1_nepo_300",100)
    
print("Done.")