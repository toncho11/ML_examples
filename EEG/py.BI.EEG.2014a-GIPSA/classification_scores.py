
from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from tqdm import tqdm

import sys
sys.path.append('.')
from braininvaders2014a.dataset import BrainInvaders2014a


from scipy.io import loadmat
import numpy as np
import mne

#from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

scr = {}

dataset = BrainInvaders2014a()

for subject in dataset.subject_list:

    #load data
    print(subject)
    sessions = dataset._get_single_subject_data(subject)
    raw = sessions['session_1']['run_1']

    # filter data and resample
    fmin = 1
    fmax = 20
    raw.filter(fmin, fmax, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'NonTarget': 1, 'Target': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
    epochs.pick_types(eeg=True)

    # get trials and labels
    X = epochs.get_data()
    y = epochs.events[:,-1]
    y = y - 1

    # cross validation
    skf = StratifiedKFold(n_splits=5)
    clf = make_pipeline(ERPCovariances(estimator='lwf', classes=[1]), MDM())
    scr[subject] = cross_val_score(clf, X, y, cv=skf, scoring = 'roc_auc').mean()

    # print results of classification
    print('subject', subject)
    print('mean AUC :', scr[subject])

#filename = './classification_scores.pkl'
#joblib.dump(scr, filename)

with open('classification_scores.txt', 'w') as the_file:
    for subject in scr.keys():
        the_file.write('subject ' + str(subject).zfill(2) + ' :' + ' {:.2f}'.format(scr[subject]) + '\n')


