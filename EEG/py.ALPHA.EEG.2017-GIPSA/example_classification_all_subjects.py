import pyriemann
print(pyriemann.__version__)
del pyriemann
#Required version of pyriemann is 0.2.7.dev
#Download from github (and replace in Anaconda if you have older version)

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyriemann.embedding import Embedding
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from alphawaves.dataset import AlphaWaves
import matplotlib.pyplot as plt
import numpy as np
import mne

"""
=============================
Classification of the trials
=============================

This example shows how to extract the epochs from the dataset of a given
subject and then classify them using Machine Learning techniques using
Riemannian Geometry. The code also creates a figure with the spectral embedding
of the epochs.

"""
# Authors: Pedro Rodrigues <pedro.rodrigues01@gmail.com>, Anton Andreev
#
# License: BSD (3-clause)

import warnings
#warnings.filterwarnings("ignore")


# define the dataset instance
dataset = AlphaWaves() # use useMontagePosition = False with recent mne versions

# get the data from subject of interest
# subject = dataset.subject_list[0]
# raw = dataset._get_single_subject_data(subject)

# # filter data and resample
# fmin = 3
# fmax = 40
# raw.filter(fmin, fmax, verbose=False)
# raw.resample(sfreq=128, verbose=False)

# # detect the events and cut the signal into epochs
# events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
# event_id = {'closed': 1, 'open': 2}
# epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
#                     verbose=False, preload=True)
# epochs.pick_types(eeg=True)

# # get trials and labels
# X = epochs.get_data()
# y = events[:, -1]

# cross validation
# skf = StratifiedKFold(n_splits=5)
# clf = make_pipeline(Covariances(estimator='lwf'), MDM())
# scr = cross_val_score(clf, X, y, cv=skf)

# # print results of classification
# print('subject', subject)
# print('mean accuracy :', scr.mean())


epochs_all_subjects = [];
label_all_subjects = [];

for subject in dataset.subject_list[0:19]: #for subjects
    
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
    
    #process all raw epochs for the selected subject 
    for i in range(0, len(epochs_subject)): 
        
        single_epoch_subject_data = epochs_subject[i]._data[0,:,:]

        #add to list
        epochs_all_subjects.append(single_epoch_subject_data)
        label_all_subjects.append(list(epochs_subject[i].event_id.values())[0]) 

X = np.array(epochs_all_subjects)
y = np.array(label_all_subjects)

skf = StratifiedKFold(n_splits=5) #20%
clf = make_pipeline(Covariances(estimator='lwf'), MDM())
scr = cross_val_score(clf, X, y, cv=skf)

# print results of classification
print('mean accuracy :', scr.mean())