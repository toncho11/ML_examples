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

"""
=============================
Classification of EGG signal from two states: eyes open and eyes closed.
Each sample of the two states is represented as an image (a recurrence plot),
next the images are classified using Deep Learning model.  
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

import warnings
#warnings.filterwarnings("ignore")

# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions


# get the data from subject of interest
#subject = dataset.subject_list[0]
#raw = dataset._get_single_subject_data(subject)

epochs_all_subjects = [];
label_all_subjects = [];

for subject in dataset.subject_list[0:5]:
    
    raw = dataset._get_single_subject_data(subject)
    
    # filter data and resample
    fmin = 3
    fmax = 40
    raw.filter(fmin, fmax, verbose=False)
    raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs_subject = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False, preload=True)
    epochs_subject.pick_types(eeg=True)
    
    #process raw epochs for the selected subject 
    for i in range(0, len(epochs_subject)):
        
        single_epoch_subject_data = epochs_subject[i]._data[0,:,:]

        #create recurrence plot of a single epoch
        rp = RecurrencePlot(threshold='point', percentage=20)
        single_epoch_subject_rp = rp.fit_transform(single_epoch_subject_data)
        print(single_epoch_subject_rp.shape)
    
        #add to list
        epochs_all_subjects.append(single_epoch_subject_rp[0,:,:])
        label_all_subjects.append(list(epochs_subject[i].event_id.values())[0] - 1 ) #from 1..2 to 0..1


#sys.exit()

train_images = np.array(epochs_all_subjects)
train_labels = np.array(label_all_subjects)

img_size = train_images[0].shape[0]

#build model

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(img_size, img_size)), #no parameter learning just transforming from 28 x 28 to 784 pixels
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2) # Each node contains a score that indicates the current image belongs to one of the 10 classes
])

model.compile(optimizer='adam', #This is how the model is updated based on the data it sees and its loss function.
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
              metrics=['accuracy']) #Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

model.fit(train_images, train_labels, epochs=10)

#training results
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)

print("Done.")