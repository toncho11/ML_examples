
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
max_epochs_per_subject = 20 #per class
    
epochs_all_subjects = [];
label_all_subjects = [];

for subject in range(1,10):

    #load data
    print("Subject =",subject)
    sessions = dataset._get_single_subject_data(subject)
    raw = sessions['session_1']['run_1']

    # filter data and resample
    fmin = 1
    fmax = 15
    raw.filter(fmin, fmax, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'NonTarget': 1, 'Target': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
    epochs.pick_types(eeg=True)

    # get trials and labels
    
    epochs_subject = epochs
    
    epochs_class_1 = 0
    epochs_class_2 = 0

    for i in range(0, len(epochs)): 
        
        label = list(epochs_subject[i].event_id.values())[0]
        
        if (label==1 and epochs_class_1 < max_epochs_per_subject) or (label==2 and epochs_class_2 < max_epochs_per_subject):
            
            single_epoch_subject_data = epochs_subject[i]._data[0,:,:]
            epochs_all_subjects.append(single_epoch_subject_data)
            label_all_subjects.append(label)
            
            if (label==1):
                epochs_class_1 = epochs_class_1 + 1
                
            if (label==2):
                epochs_class_2 = epochs_class_2 + 1
        
X = np.array(epochs_all_subjects)
y = np.array(label_all_subjects)

print(X.shape)

print("Classification . . . ")
    
# cross validation
skf = StratifiedKFold(n_splits=5)
clf = make_pipeline(ERPCovariances(estimator='lwf', classes=[1]), MDM())
#scr[subject] = cross_val_score(clf, X, y, cv=skf, scoring = 'roc_auc').mean()
scr = cross_val_score(clf, X, y, cv=skf)
print('mean accuracy :', scr.mean())

