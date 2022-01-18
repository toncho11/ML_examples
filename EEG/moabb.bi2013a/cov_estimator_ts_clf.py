#sklearn 23.1 used
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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

#datasets = [bi2013a()] # , EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
datasets = [EPFLP300()]
paradigm = P300()

le = LabelEncoder()

cov_estimator = XdawnCovariances(classes=[1], estimator='oas', xdawn_estimator='oas')
ts = TangentSpace()
clf = LogisticRegression(solver="liblinear")
pipeline = make_pipeline(cov_estimator, ts, clf)

n_test_subjects_max = 10;
n_test_samples_max = 1000; #max per class and per subject
    
scorer = make_scorer(balanced_accuracy_score)

for dataset in datasets:
    print("Total subjects in dataset:", len(dataset.subject_list))

    X=[]
    y=[]
    n_test_subjects = min(n_test_subjects_max,len(dataset.subject_list))
    
    #load data
    #iterate over the subjects
    #for source_i, source in enumerate(dataset.subject_list[4:9]): #EPFLP300 to get only the healthy subjects
    for source_i, source in enumerate(dataset.subject_list[:n_test_subjects_max]):
        print("Loading subject:", source_i, " id = ", source)
        #X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:n_test_subjects_max]) #dataset.subject_list[:10])#
        
        #get single subject
        X1, y1, _ = paradigm.get_data(dataset=dataset, subjects=[source]) 
        
        y1 = le.fit_transform(y1)
        
        #set an upper limit for the samples (without considering target or non target)
        # n_samples = X1.shape[0]
        # if (n_samples > 1000):
        #     print("Samples reduced to: ", n_test_samples_max)
        #     X1 = X1[:n_test_samples_max,:,:]
        #     y1 = y1[:n_test_samples_max]
        ##end setting upper limit
            
        #set an upper limit per class (allows both classes to be more equally presented in training set)
         
        epochs_class_1 = 0
        epochs_class_2 = 0
        index_label1 = [];
        index_label2 = [];
        
        #get only the required number of samples
        for idx,val in enumerate(y1):
            if (val == 0 and epochs_class_1 < n_test_samples_max):
                index_label1.append(idx)
                epochs_class_1 = epochs_class_1 + 1
            elif (val == 1 and epochs_class_2 < n_test_samples_max):
                index_label2.append(idx)
                epochs_class_2 = epochs_class_2 + 1

        X1 = X1[index_label1 + index_label2,:,:]
        y1 = y1[index_label1 + index_label2]
        #end setting upper limit
        
        print("Total class target samples available for subject: ", sum(y1))
        print("Total class non-target samples available for subject: ", len(y1) - sum(y1))
        
        print("Finished loading subject:", source_i)
        print("Samples = ", X1.shape[0], " | Electrodes = ", X1.shape[1], " | Sample length=", X1.shape[2])
        
        #append data from current subject X1 and y1 to X and y
        if (X == []):
            X = X1
            y = y1
        else:
            X = np.concatenate((X,X1),axis=0)
            y = np.concatenate((y,y1),axis=0)
    
    print("Total class target samples available: ", sum(y))
    print("Total class non-target samples available: ", len(y) - sum(y))
    
    print("Starting classification:")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer, n_jobs=-1)
    print(scores)
    print("Mean score: ",np.mean(scores))
