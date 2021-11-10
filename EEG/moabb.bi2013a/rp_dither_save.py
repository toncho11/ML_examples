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

import Dither
import os
import glob

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

datasets = [bi2013a()] # , EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
paradigm = P300()

le = LabelEncoder()

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
    
    #percents = np.percentile(X_dist,percentage)
    
    #X_rp = X_dist < percents
    
    out = Dither.dither(X_dist, 'floyd-steinberg', resize=False)
    
    return out#X_rp

def CreateData(m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject):
    
    folder = "D:\Work\ML_examples\EEG\moabb.bi2013a\data"
    
    folder = folder + "\\rp_dither_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) 
    
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write rp image data:")
    
    
    # for subject in range(1,n_subjects+1):
    
    #     #load data
    #     print("Subject =",subject)
    #     sessions = dataset._get_single_subject_data(subject)
    #     raw = sessions['session_1']['run_1']
    
    #     # filter data and resample
    #     fmin = filter_fmin
    #     fmax = filter_fmax
    #     raw.filter(fmin, fmax, verbose=False)
    
    #     # detect the events and cut the signal into epochs
    #     events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    #     event_id = {'NonTarget': 1, 'Target': 2}
    #     epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=0.8, baseline=None, verbose=False, preload=True)
    #     epochs.pick_types(eeg=True)
    
    #     # get trials and labels
        
    #     epochs_subject = epochs
        
    #     epochs_class_1 = 0
    #     epochs_class_2 = 0
        
    #     for i in range(0, len(epochs)): 
            
    #         single_epoch_subject_data = epochs_subject[i]._data[0,:,:]
    
    #         label = list(epochs_subject[i].event_id.values())[0]-1 #sigmoid requires that labels are [0..1]
            
    #         #save
    #         if (label==0 and epochs_class_1 < max_epochs_per_subject) or (label==1 and epochs_class_2 < max_epochs_per_subject):
    
    #             single_epoch_subject_rp = multivariateRP(single_epoch_subject_data, electrodes, m, tau, percentage)
                
    #             filename = "subject_" + str(subject-1) + "_rp_label_" + str(label) + "_epoch_" + str(i)
    #             full_filename = folder + "\\" + filename
                
    #             print("Saving: " + full_filename)
    #             #plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
    #             np.save(full_filename, single_epoch_subject_rp)
                
    #             if (label==0):
    #                 epochs_class_1 = epochs_class_1 + 1
                    
    #             if (label==1):
    #                 epochs_class_2 = epochs_class_2 + 1

# for dataset in datasets:
#     for source_i, source in enumerate(dataset.subject_list):
#         X, y, _ = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:2]) #dataset.subject_list[:10])#
#         y = le.fit_transform(y)

