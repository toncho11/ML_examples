import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

import tensorflow as tf
from tensorflow import keras

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from alphawaves.dataset import AlphaWaves

import mne

"""
=============================
Example on how to classify using Deep Learning with Keras
=============================

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5
TensorFlow 2.3.0
Keras is installed as part of TensorFlow 2

"""
# Authors: Anton Andreev
#
# License: BSD (3-clause)

import warnings
#warnings.filterwarnings("ignore")


# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions

# get the data from subject of interest
subject = dataset.subject_list[0]
raw = dataset._get_single_subject_data(subject)

# filter data and resample
fmin = 3
fmax = 40
raw.filter(fmin, fmax, verbose=False)
raw.resample(sfreq=128, verbose=False)

# detect the events and cut the signal into epochs
events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
event_id = {'closed': 1, 'open': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                    verbose=False, preload=True)
epochs.pick_types(eeg=True)

# get trials and labels
X = epochs.get_data()
y = events[:, -1]

# cross validation
skf = StratifiedKFold(n_splits=5)

print("Done.")