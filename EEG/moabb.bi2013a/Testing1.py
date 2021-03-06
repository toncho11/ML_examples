# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:45:03 2022

@author: antona
"""

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
from tensorflow.keras import optimizers

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from PIL import Image as im

import mne
#from pyts.image import RecurrencePlot
import gc
import os
from sklearn import svm
from sklearn.model_selection import train_test_split

import cv2

#disable GPU because the memory of the GPU is not enough or some bug
disableGPU = True
if (disableGPU):
    print('GPU is disasabled on purpose')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

# config = tf.ConfigProto(
#         device_count = {'GPU': 0}
#     )
# sess = tf.Session(config=config)

#it works with cudnn-11.3-windows-x64-v8.2.1.32 (Cuda 11.3/11.5 and cuddn 8.2)

# def limitgpu(maxmem):
# 	gpus = tf.config.list_physical_devices('GPU')
# 	if gpus:
# 		# Restrict TensorFlow to only allocate a fraction of GPU memory
# 		try:
# 			for gpu in gpus:
# 				tf.config.experimental.set_virtual_device_configuration(gpu,
# 						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
# 		except RuntimeError as e:
# 			# Virtual devices must be set before GPUs have been initialized
# 			print(e)


#limitgpu(3000)
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

def PlotTrainValidAccuracy(epochs, history):
    plt.clf()
    acc = history.history_dict['acc']
    val_acc = history.history_dict['val_accuracy']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training vs Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def LoadImages(folder, n_max_subjects, n_max_samples):

    epochs_all_subjects = []
    label_all_subjects = []
    
    samples_class1 = np.zeros(n_max_subjects) #for each subject
    samples_class2 = np.zeros(n_max_subjects)
    
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

                if (label == 0 and samples_class1[subject] < n_max_samples) or (label == 1 and samples_class2[subject] < n_max_samples):
                    images_loaded = images_loaded + 1

                    if label == 0:
                        samples_class1[subject] = samples_class1[subject] + 1
                    elif label == 1:
                        samples_class2[subject] = samples_class2[subject] + 1

                    rp_image=np.load(os.path.join(folder, filename))

                    epochs_all_subjects.append(rp_image)

                    label_all_subjects.append(label)
            
        else:
            continue
    
    print("Images loaded: ", images_loaded)
    return epochs_all_subjects, label_all_subjects

def Model1(img_size1, img_size2):
    
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
    return model

def Model2(img_size1, img_size2):
    
    t = 4
    model = Sequential()
    model.add(Conv2D(32, (t, t), input_shape=(img_size1,img_size2,1)))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(t/2, t/2)))
      
    model.add(Conv2D(64, (t, t)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(t/2, t/2)))
      
    model.add(Conv2D(128, (t, t)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(t/2, t/2)))
    
    model.add(Conv2D(128, (t, t)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(t/2, t/2)))
      
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['accuracy'])
    return model

def Evaluate(model, x_test, y_test):
    
    actual = y_test
    print("=====================================================")
    
    results = model.evaluate(x_test, y_test)
    print("Evaluate on test data (never seen): test acc:", results[1])
    
    y_pred =  model.predict(x_test)
    y_pred_bin = np.round_(y_pred).astype(int)
    ba = balanced_accuracy_score(actual, y_pred_bin)
    print("Evaluate on test data (never seen): balanced accuracy:", ba)

def calcDist(i1, i2):
    return np.sum((i1-i2)**2)

def calcDistManhattan(i1, i2):
    return np.sum(abs((i1-i2)))

def calculateDistance(i1, i2):
    return calcDist(i1, i2)
    
def ProcessFolder(epochs_all_subjects, label_all_subjects):
    
    #build model
    
    #to prevent overfitting
    # - adjust the number of epochs
    # - smaller model
    # - regularizing L1, L2 or both
    # - dropout layers

    img_size1, img_size2 = np.array(epochs_all_subjects[0]).shape
    print("Image size: ", img_size1, img_size2)
    
    model = Model1(img_size1, img_size2)

    iterations = 1
    average_classification = 0;
    test_n = 400
        
    for i in range(iterations):
        
        print("Iteration: ",i)
        indices = np.arange(len(epochs_all_subjects))
        np.random.shuffle(indices)
        all_images_shuffled = np.array(epochs_all_subjects)[indices]
        labels_shuffled = np.array(label_all_subjects)[indices]
            
        #shuffle
        for s in range(0,20):
            #print(s)
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


        epochs = 130;
        
        #model.fit(X_train, y_train, epochs=5, validation_data = (X_test,y_test) ) #validation_data=(X_test,y_test)
        #history = model.fit(np.array(epochs_all_subjects)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        #history = model.fit(np.array(all_images_shuffled)[:, :, :, np.newaxis],  np.array(labels_shuffled), epochs=epochs, validation_split=0.2 )
        
        data_to_process = np.array(all_images_shuffled).astype('float32') #[:, :, :, np.newaxis]
        data_to_process = data_to_process.reshape(data_to_process.shape[0],data_to_process.shape[1],data_to_process.shape[2],1)
        print(data_to_process.shape)
        
        l1 = []
        l2 = []
        for k in range(len(data_to_process)):
            if (labels_shuffled[k] == 0):
                l1.append(data_to_process[k])
            elif (labels_shuffled[k] == 1): 
                l2.append(data_to_process[k])
        
        #prepare data
        test_x = data_to_process[0:test_n]
        test_y = labels_shuffled[0:test_n]
        data_to_process = data_to_process[test_n:]
        labels_shuffled = labels_shuffled[test_n:]
        
        #calculate average images
        imave1 = np.average(l1,axis=0) #non target
        imave2 = np.average(l2,axis=0) #target
        
        norm_image1 = cv2.normalize(imave1[:,:,0], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_image2 = cv2.normalize(imave2[:,:,0], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        
        #sample = data_to_process[4]   
        #plt.imshow(sample, cmap = plt.cm.binary, origin='lower')
        img1Average = im.fromarray(norm_image1)
        img2Average = im.fromarray(norm_image2)
        
        img1AverageRotated = img1Average#.rotate(-45)
        img2AverageRotated = img2Average#.rotate(-45)
        
        #data.save('gfg_dummy_pic.png')
        #data.save('gfg_dummy_pic.png')
        img1AverageRotated.convert("L").save('c:\\temp\\img1AverageRotated.png')
        img2AverageRotated.convert("L").save('c:\\temp\\img2AverageRotated.png')
        #end calculate average images
       
        
        print("Test: Class non-target samples: ", len(test_y) - sum(test_y))
        print("Test: Class target samples: ", sum(test_y))
        
        print("Train: Class non-target samples: ", len(labels_shuffled) - sum(labels_shuffled))
        print("Train: Class target samples: ", sum(labels_shuffled))
        
        accuracy = 0;
        #train model
        
        histogram1, bin_edges1 = np.histogram(imave1, bins=256, range=(0, 1))
        histogram2, bin_edges2 = np.histogram(imave2, bins=256, range=(0, 1))
        
        class0 = 0
        class1 = 0
        pred=[]
        
        for k in range(len(test_x)):
        
            img = test_x[k,:,:]
            #hist, bin_edges = np.histogram(img, bins=256, range=(0, 1))
            
            diff1 =  (img - imave1)
            diff2 =  (img - imave2)
            
            m_norm1 = np.linalg.norm(diff1)
            m_norm2 = np.linalg.norm(diff2)
            
            if (m_norm1 < m_norm2):
                pred.append(0)
            else:
                pred.append(1)
                
            # if (m_norm1 > m_norm2 and test_y[k] == 0):
            #     accuracy = accuracy + 1
            #     class0 = class0 + 1
            
            # else:#if (m_norm1 <= m_norm2 and test_y[k] == 1):
            #     accuracy = accuracy + 1
            #     class1 = class1 + 1
        
        #print("Class0: ", class0) #non target
        #print("Class1: ", class1) #target
        #print(pred)
        
        #print("Accuracy: ", accuracy / len(test_x))
        
        ba = balanced_accuracy_score(test_y, pred)
        print("Evaluate on test data (never seen): balanced accuracy:", ba)
        
        #Evaluate(model, test_x, test_y)
        #return history.history['val_accuracy'][epochs-1]
        
        
#print("Test data:================================================================================================================")

#data_folder="D:\Work\ML_examples\EEG\moabb.bi2013a\data"
#data_folder="H:\data"
data_folder="C:\Temp\data"
#data_folder="h:\data"
#configure tensor flow to avoid GPU out memory error
#https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow/60558547#60558547



# results = []
# max_folder = 20;
# i = 0;

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

#folder = data_folder + "\\rp_dither_m_5_tau_40_f1_1_f2_20_el_4_nsub_12_per_-1_nepo_300" #0.67
#folder = data_folder + "\\rp_m_5_tau_40_f1_1_f2_24_el_8_nsub_16_per_20_nepo_200"
#folder = data_folder + "\\rp_m_6_tau_40_f1_1_f2_24_el_8_nsub_3_per_20_nepo_50" 
folder = data_folder + "\\rp_m_7_tau_20_f1_1_f2_24_el_all_nsub_5_per_20_nepo_800_set_bi2013a_xdawn_yes_dither" 
#rp_m_5_tau_30_f1_1_f2_24_el_8_nsub_10_per_20_nepo_800_set_BNCI2015003_xdawn_yes
epochs_all_subjects, label_all_subjects = LoadImages(folder, 10, 10000)
ProcessFolder(epochs_all_subjects, label_all_subjects)
    
print("Done.")