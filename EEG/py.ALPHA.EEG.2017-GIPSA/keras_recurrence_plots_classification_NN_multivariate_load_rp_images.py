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
length_s = 29 #max=19
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


train_images = np.array(train_epochs_all_subjects)[:, :, :, np.newaxis] # we add an extra axis as required by keras
#train_images = np.array(train_epochs_all_subjects)[:, 200:280, 200:280, np.newaxis]
train_labels = np.array(train_label_all_subjects)


img_size = train_images[0].shape[1]
n = train_images.shape[0]
print(img_size)


#NN
train_images = train_images[:,:,:,-1]
#train_images = train_images.reshape((n,img_size * img_size))

model = Sequential()
model.add( Dense(512, activation='relu', input_shape=(img_size,img_size) ))
model.add( Dense(384, activation='relu' ))
model.add( Dense(128, activation='relu' ))
model.add( Dense(32, activation='relu' ))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=200, validation_split=0.2, shuffle=True)

# #SVM
# clf_svm = svm.SVC()
# clf = svm.SVC(kernel='linear', C=1).fit(X_train(-1,:,:,-1), y_train)
# print("SVM on test: ", clf.score(X_test, y_test))

#print("Testing:")
#training results
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy on unseen data:', test_acc)

print("Done.")