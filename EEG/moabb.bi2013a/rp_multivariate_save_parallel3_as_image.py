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

import sys
sys.path.append('C:\\Work\\PythonCode\\ML_examples\\EEG\\P300Analysis\\')
from DatasetHelper import *

import Dither
import os
import glob
import time

from joblib import Parallel, delayed
from multiprocessing import Process

from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

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

# generates interplated samples from the original samples
def GenerateInterpolated(X,y, selected_class, every_ith_sample):
    
    print ("Generating interpolated samples...")
    
    #every_ith_sample = 8 #default 10
    
    Xint = X.copy()
    epoch_length = X.shape[2]
    channels     = X.shape[1]
    indicesSelectedClass = []
    
    for i in range(0, len(y)):
        if y[i] == selected_class:
            for j in range(0,epoch_length,every_ith_sample):
                if j - 1 > 0 and j + 1 < epoch_length:
                    for m in range(0,channels):
                        #modify it
                        Xint[i,m,j] = (X[i,m,j-1] + X[i,m,j+1]) / 2
            indicesSelectedClass.append(i)
    
    print("Interpolated samples generated:", len(indicesSelectedClass))
    return Xint[indicesSelectedClass,:,:], np.repeat(selected_class,len(indicesSelectedClass))

def ProcessSamples(samples, X, y, folder, subject, m, tau , electrodes, percentage):

    for sample_i in samples:
        print("Process Sample:",sample_i)
        label  = y[sample_i]
        sample = X[sample_i]
        
        #augment sample if P300
    
        single_epoch_subject_rp = multivariateRP(sample, electrodes, m, tau, percentage)
    
        filename = "subject_" + str(subject - 1) + "_rp_label_" + str(label) + "_epoch_" + str(sample_i)
        full_filename = folder + "\\" + filename
    
        print("Saving: " + full_filename)
        # plt.imshow(single_epoch_subject_rp, cmap = plt.cm.binary)
        #np.save(full_filename, single_epoch_subject_rp)
        
        #save as image
        im = Image.fromarray(single_epoch_subject_rp)
        im.save(full_filename + ".jpeg")

def CreateData(dataset, m, tau , filter_fmin, filter_fmax, electrodes, n_subjects, percentage, max_epochs_per_subject, data_augment):
    
    folder = "C:\\Work\PythonCode\\ML_examples\\EEG\\moabb.bi2013a\\data"
    #folder = "h:\\data"
    #folder = "h:\\data"

    folder = folder + "\\rp_m_" + str(m) + "_tau_" + str(tau) + "_f1_"+str(filter_fmin) + "_f2_"+ str(filter_fmax) + "_el_" + str(len(electrodes)) + "_nsub_" + str(n_subjects) + "_per_" + str(percentage) + "_nepo_" + str(max_epochs_per_subject) + "_set_" + dataset.__class__.__name__ + "_da_" + ("T" if data_augment==True else "F") + "_as_image"
   
    print(folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Clean data:")
    
    files = glob.glob(folder + "\\*")
    for f in files:
        if f.endswith(".npy"):
            os.remove(f)
        
    print("Write rp image data:")
    
      
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

        #start data augmentation
        if (data_augment):
            #Addding samples by revmoving some data and replacing it with interplated data
            P300Class = 1 #only designed for 1 now (P300)
            original_n_count = X.shape[0]
            
            #augmentation 1
            P300ClassCount = sum(y)
            #NonTargetCount = len(y) - P300ClassCount
            
            X_interpolated1, y_interpolated1 = GenerateInterpolated(X, y, P300Class, 7)
            if (P300ClassCount != X_interpolated1.shape[0]):
                raise Exception("Problem with interpolated data 1!")
                
            X_interpolated2, y_interpolated2 = GenerateInterpolated(X, y, P300Class, 12)
            if (P300ClassCount != X_interpolated2.shape[0]):
                raise Exception("Problem with interpolated data 2!")
                
            X = np.concatenate((X, X_interpolated1), axis=0)
            y = np.concatenate((y, y_interpolated1), axis=0)
            
            print("Class target samples (after data augmentation): ", sum(y))
            print("Class non-target samples (after data augmentation): ", len(y) - sum(y))
            
            X = np.concatenate((X, X_interpolated2), axis=0)
            y = np.concatenate((y, y_interpolated2), axis=0)
            
            print("Class target samples (after data augmentation): ", sum(y))
            print("Class non-target samples (after data augmentation): ", len(y) - sum(y))
            
            if (sum(y) % P300ClassCount != 0):
                raise Exception("Problem with interpolated data count!")
            
            
        #end data augmentation
        
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
        
        print("Selected non-target samples: ", epochs_class_1)
        print("Selected target samples: ",  epochs_class_2)
        
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

    #name,    electrodes,   subjects
    #bi2013a	      16	24 (normal)                    
    #bi2014a    	  16	64 (usually low performance)
    #BNCI2014009	  16	10 (usually high performance)
    #BNCI2014008	   8	 8
    #BNCI2015003	   8	10
    #bi2015a          32    43
    #bi2015b          32    44
    db = BNCI2014009()#BNCI2014008()
    
    db_bame = GetDatasetNameAsString(db)
    
    electrodes = [GetElectrodeByName(db_bame,"Fz"), GetElectrodeByName(db_bame,"Cz"), GetElectrodeByName(db_bame,"Pz"), GetElectrodeByName(db_bame,"Oz") ]
    
    #CreateData(db, 5, 30 , f1 ,f2 , electrodes, 1, 20, 600)
    #CreateData(db, 5, 30 , f1 ,f2 , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 1, 20, 600)
    
    #CreateData(db, 5, 30 , f1 ,f2 , electrodes, 3, 20, 400)
    #CreateData(db, 5, 30 , f1 ,f2 , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 3, 20, 400)
    
    #CreateData(db, 5, 30 , f1 ,f2 , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 10, 20, 400)
    #CreateData(db, 5, 30 , f1 ,f2 , electrodes, 10, 20, 400)
    
    #CreateData(db, 5, 30 , f1 ,f2 , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 10, 20, 800)
    CreateData(db, 5, 30 , f1 ,f2 , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 10, 20, 2000, True)
    
    end = time.time()
    print("Elapsed time (in seconds):",end - start)
    
    
