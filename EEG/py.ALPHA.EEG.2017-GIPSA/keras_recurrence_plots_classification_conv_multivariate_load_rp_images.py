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

from alphawaves.dataset import AlphaWaves

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

# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions

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
folder = "D:\Work\ML_examples\EEG\py.ALPHA.EEG.2017-GIPSA\multivariate_rp_images"

train_epochs_all_subjects = [];
train_label_all_subjects = [];

test_epochs_all_subjects = [];
test_label_all_subjects = [];

print("Train data:")

images_loaded = 0
for filename in os.listdir(folder):
    if filename.endswith(".npy"): 
        
        print(os.path.join(folder, filename))
        base_name = os.path.basename(filename)
        
        parts = base_name.split("_")
        #print(parts)
        label = int(parts[4].split(".")[0])
        subject = int(parts[1])
        print("Subject: ", subject, " Label: ", label)
        
        if (subject < n_train_subjects):
            images_loaded = images_loaded + 1
            rp_image=np.load(os.path.join(folder, filename))
            train_epochs_all_subjects.append(rp_image)
            train_label_all_subjects.append(label)
        
    else:
        continue

print("Train images loaded: ", images_loaded)

print("Test data:================================================================================================================")

# images_loaded = 0
# for filename in os.listdir(folder):
#     if filename.endswith(".npy"): 
#         print(os.path.join(folder, filename))
#         base_name = os.path.basename(filename)
        
#         parts = base_name.split("_")
#         #print(parts)
#         label = int(parts[4].split(".")[0])
#         subject = int(parts[1])
#         print("Subject: ", subject, " Label: ", label)
        
#         if (subject >= n_train_subjects):
#             images_loaded = images_loaded + 1
#             rp_image=np.load(os.path.join(folder, filename))
#             test_epochs_all_subjects.append(rp_image)
#             test_label_all_subjects.append(label)

# print("Test images loaded: ", images_loaded)

#train_images = np.array(train_epochs_all_subjects)[:, :, :, np.newaxis] # we add an extra axis as required by keras
#train_images = np.array(train_epochs_all_subjects)[:, 0:300, 0:300, np.newaxis] # we add an extra axis as required by keras

train_images1 = []
train_images2 = []

#produce some test results
for i in range(0,len(train_label_all_subjects)):
    if train_label_all_subjects[i] == 0: # 0 eyes closed = alpha
        train_images1.append(train_epochs_all_subjects[i])#[200:280, 200:280]
    else:
        train_images2.append(train_epochs_all_subjects[i])#[200:280, 200:280]
        
imave1 = np.average(train_images1,axis=0) #eyes closed, alpha high
imave2 = np.average(train_images2,axis=0) #eyes opened, alpha low

#train_images = np.array(train_epochs_all_subjects)[:, 110:200, 110:200, np.newaxis] # we add an extra axis as required by keras
train_images = np.array(train_epochs_all_subjects)[:, 110:200, 150:200, np.newaxis] # 0.73 # we add an extra axis as required by keras
#train_images = np.array(train_epochs_all_subjects)[:, 200:280, 200:280, np.newaxis] # we add an extra axis as required by keras

# train_images1 = np.array(train_epochs_all_subjects)[:, 200:290, 200:290]
# train_images2 = np.array(train_epochs_all_subjects)[:, 480:570, 480:570]

# train_images = []
# for i in range(0,len(train_images1)):
#     train_images.append(np.concatenate((train_images1[i,],train_images2[i,])))
# train_images = np.array(train_images)[:, :, :, np.newaxis]

train_labels = np.array(train_label_all_subjects)

#test_images = np.array(test_epochs_all_subjects)[:, :, :, np.newaxis]
#test_labels = np.array(test_label_all_subjects)


#X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.4, random_state=0)


img_size1 = train_images[0].shape[0]
img_size2 = train_images[0].shape[1]
print(img_size1, img_size2)

#build model

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

# compile model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', #adam, rmsprop
              metrics=['accuracy'])

#model.fit(train_images, train_labels, epochs=8)
#model.fit(train_images, train_labels, epochs=8, validation_split=0.2, shuffle=True)
#model.fit(train_images, train_labels, epochs=20, shuffle=True)
model.fit(train_images, train_labels, epochs=30, validation_split=0.2, shuffle=True)

# #SVM
# clf_svm = svm.SVC()
# clf = svm.SVC(kernel='linear', C=1).fit(X_train(-1,:,:,-1), y_train)
# print("SVM on test: ", clf.score(X_test, y_test))

#print("Testing:")
#training results
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy on unseen data:', test_acc)

print("Done.")