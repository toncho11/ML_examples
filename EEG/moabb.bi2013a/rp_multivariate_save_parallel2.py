import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a #, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

import numpy as np

from sklearn.preprocessing import LabelEncoder

import Dither
import os
import glob

from joblib import Parallel, delayed
from multiprocessing import Manager, Value
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

datasets = [bi2013a()] # , EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
paradigm = P300()

le = LabelEncoder()

#bi2013a: FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2

print("Start declaring Parallel manager")
if __name__ == '__main__':
    epochs_class_1 = Value('i', 0)
    epochs_class_2 = Value('i', 0)

lock = threading.Lock()
print("End declaring Parallel manager")
        
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
    
    percents = np.percentile(X_dist,percentage)
    
    X_rp = X_dist < percents
    
    #out = Dither.dither(X_dist, 'floyd-steinberg', resize=False)
    
    return X_rp #out

def Process(sample_i, sample, y, folder, subject, m, tau , electrodes, percentage, max_epochs_per_subject):

    print("In Process")
    label = y[sample_i]

   
    #lock = threading.Lock()
    lock.acquire()
    
    save = False;
    
    if (label==0 and epochs_class_1.value < max_epochs_per_subject) or (label==1 and epochs_class_2.value < max_epochs_per_subject):
        
        if (label == 0):
            epochs_class_1.value = epochs_class_1.value + 1
     
        if (label == 1):
            epochs_class_2.value = epochs_class_2.value + 1
            
        save = True
    
    lock.release()
    
    if (save):
        
        single_epoch_subject_rp = multivariateRP(sample, electrodes, m, tau, percentage)
        filename = "subject_" + str(subject - 1) + "_rp_label_" + str(label) + "_epoch_" + str(sample_i)
        full_filename = folder + "\\" + filename
    
        print("Saving: " + full_filename)
        # plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
        np.save(full_filename, single_epoch_subject_rp)

def CreateData(m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject):
    
    folder = "C:\\Work\PythonCode\\ML_examples\\EEG\\moabb.bi2013a\\data"
    #folder = "r:\\data"

    folder = folder + "\\rp_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) 
    
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write rp image data:")
    

    
    for dataset in datasets:
        
        for subject_i, subject in enumerate(dataset.subject_list[0:n_subjects]):
            

            lock.acquire()
            epochs_class_1.value = 0
            epochs_class_2.value = 0
            lock.release();
            print("Loading subject:" , subject)  
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            y = le.fit_transform(y)
            print(X.shape) 
            #0 NonTarget
            #1 Target       
            print("Class target samples: ", sum(y))
            print("Class non-target samples: ", len(y) - sum(y))
            

            #def Process(sample_i, X, y, folder, subject, m, tau, electrodes, percentage):
            Parallel(n_jobs=24, backend = "threading")(delayed(Process)(sample_i, sample, y, folder, subject, m, tau, electrodes, percentage, max_epochs_per_subject) for sample_i, sample in enumerate(X))

        
f1 = paradigm.filters[0][0]
f2 = paradigm.filters[0][1]

start = time.time()
CreateData(5,40,f1,f2,[8,9,10,11,12,13,14,15],2,20,20)
end = time.time()
print("Elapsed time (in seconds):",end - start)