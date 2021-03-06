import matplotlib.pyplot as plt
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
import time

from joblib import Parallel, delayed
from multiprocessing import Process
from scipy.spatial.distance import cdist, mahalanobis

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

datasets = [BNCI2014008()] # , bi2013a(), EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
paradigm = P300()

le = LabelEncoder()

#https://www.researchgate.net/figure/Common-electrode-setup-for-P300-spellers-according-to-8-Eight-EEG-electrodes-are_fig1_221583051
#Common electrode setup for P300 spellers according to [8]. 
# Eight EEG electrodes are placed at Fz, Cz, P3, Pz, P4, PO7, Oz and PO8. [3,6,9,10,11,14,15,16] 
#bi2013a: FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2

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
    
    img = X_traj
    V = np.cov(np.array(img))
    IV_ = np.linalg.inv(V)

    #make an ij loop to claculate and store all distances 
    #print(mahalanobis(array_1, array_2, IV))

    # for i in range(0,img.shape[0]): #i is the vector
    #     for j in range(0,img.shape[1]):
    #         v1 = img[i,:]
    #         v2 = img[:,j] #changed the way the vectros is taken
    #         print(v1.ndim)
    #         ff = np.expand_dims(v1, axis=0)
    #         cdist(ff,ff,'mahalanobis',IV=IV)

    print("Calculating distance")
    #result = cdist(img[3:,],img[5:,],'mahalanobis')

    result1 = cdist(img,img,'mahalanobis', VI = IV_)
    result1[np.isnan(result1)] = 0
    
    return result1

def ProcessSamples(samples, X, y, folder, subject, m, tau , electrodes, percentage):

    for sample_i in samples:
        print("Process Sample:",sample_i)
        label = y[sample_i]
        sample = X[sample_i]
    
        single_epoch_subject_rp = multivariateRP(sample, electrodes, m, tau, percentage)
    
        filename = "subject_" + str(subject - 1) + "_rp_label_" + str(label) + "_epoch_" + str(sample_i)
        full_filename = folder + "\\" + filename
    
        print("Saving: " + full_filename)
        # plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
        np.save(full_filename, single_epoch_subject_rp)

def CreateData(dataset, m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject):
    
    #folder = "C:\\Work\PythonCode\\ML_examples\\EEG\\moabb.bi2013a\\data"
    #folder = "h:\\data"
    folder = "h:\\data"
    #folder = "c:\\temp\\data"

    folder = folder + "\\rp_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) + "_set_" + dataset.__class__.__name__ 
    
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write rp image data:")
    
    
    #for dataset in datasets:
        
    for subject_i, subject in enumerate(dataset.subject_list[0:n_subjects]):
        
        epochs_class_1 = 0
        epochs_class_2 = 0
        
        print("Loading subject:" , subject)  
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
        y = le.fit_transform(y)
        print(X.shape) 
        #0 NonTarget
        #1 Target       
        print("Class target samples: ", sum(y))
        print("Class non-target samples: ", len(y) - sum(y))

        index_label1 = [];
        index_label2 = [];
        
        #get only the required number of samples
        for idx,val in enumerate(y):
            if (val == 0 and epochs_class_1 < max_epochs_per_subject):
                index_label1.append(idx)
                epochs_class_1 = epochs_class_1 + 1
            elif (val == 1 and epochs_class_2 < max_epochs_per_subject):
                index_label2.append(idx)
                epochs_class_2 = epochs_class_2 + 1
        
        print("Selected data target samples: ", epochs_class_1)
        print("Selected non-target samples: ",  epochs_class_2)
        
        print("Processing")
        n_jobs = 9
        processes = [None] * n_jobs            
        i=0          
        parallel = False
        
        if (parallel):
        
            print("Starting parallel processes")
            
            for section in np.array_split(index_label1 + index_label2 , n_jobs):
                processes[i] = Process(target=ProcessSamples,args=(section, X, y, folder, subject, m, tau, electrodes, percentage))
                processes[i].start()
                print(i)
                i = i + 1
            
            print("Setting threads to join:")
            for p in processes:
                 p.join()
        else:
            ProcessSamples(index_label1 + index_label2, X, y, folder, subject, m, tau, electrodes, percentage)

if __name__ == '__main__':

    start = time.time()
    f1 = paradigm.filters[0][0]
    f2 = paradigm.filters[0][1]

    #CreateData(5,40,f1,f2,[8,9,10,11,12,13,14,15],16,20,200) #standard
    #CreateData(5,30,f1,f2,[9,10,11,13,14,15],1,20,800) #a different electrode set
    #CreateData(5,30,f1,f2,[9,10,11,13,14,15],1,20,800) #a different electrode set
    #CreateData(5,30,f1,f2,[6,7],1,20,800) #a different electrode set
    #CreateData(BNCI2014008(), 5,30,f1,f2,[4,5,7],3,20,800) #good results
    
    #bi2013a: FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2
    #CreateData(bi2013a(), 5,30,f1,f2,[13,14,15], 5, 20, 800)
    
    CreateData(BNCI2014008(), 5,30,f1,f2,[4,5,7], 2, 20, 400)
    
    end = time.time()
    print("Elapsed time (in seconds):",end - start)
    
    
