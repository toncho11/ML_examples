# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:58:24 2024

@author: antona

Result:
    0
    11
    11
    11
    10
    10
    10
    10
    11
    11
    0
    Duration 1: 3090.9477999666706
    Total iterations 1: 95
    0
    6
    6
    7
    7
    11
    7
    7
    6
    6
    0
    Duration 2: 2272.8909999132156
    Total iterations 2: 63
    Time difference in %: 26.47 %
    Iteration difference in %: 33.68 %
"""

from pyriemann.datasets import generate_random_spd_matrix
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import warnings

from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance, mean_power, mean_logeuclid
from pyriemann.utils.distance import distance
from pyriemann.tangentspace import FGDA, TangentSpace
from pyriemann.utils.distance import distance_euclid
from scipy.stats import zscore
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import copy
from enchanced_mdm_mf_tools import mean_power_custom, distance_custom, power_distance
from time import perf_counter_ns,perf_counter
from random import randint

power_means11 = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]

power_list = power_means11

def calculate_mean(X, y, p, sample_weight, reuse_previous_mean, covmeans):
    
    means_p = {}
    
    pos = power_list.index(p)
    itr_all = 0
    
    for ll in [0,1]:
        
        init = None
        
        #use the mean from the previous position in the power list
        if reuse_previous_mean:
            
            if pos>0:
                prev_p = pos-1
                init = covmeans[prev_p][ll]
                #print(prev_p)
                #print("using prev mean from the power list")
            
        means_p[ll], itr = mean_power_custom(
            X[y == ll],
            p,
            sample_weight=sample_weight[y == ll],
            zeta = 1e-07,
            init = init
        )
        itr_all = itr_all + itr
        
    covmeans[pos] = means_p
        
    return means_p,itr
        
#############################################################################
n = 2000

X = np.array([generate_random_spd_matrix(10) for i in range(0,n)])

y = np.array([randint(0, 1) for i in range(0,n)]) 

sample_weight = np.ones(X.shape[0])

# Test 1 - reuse previous mean
covmeans = {}

time_start = perf_counter()
itr_total1 = 0
for p in power_list:
    mean,itr = calculate_mean(X, y, p, sample_weight, True , covmeans)
    print(itr)
    itr_total1 = itr_total1 + itr
time_end = perf_counter()
time_duration1 = time_end - time_start
print("Duration 1:", time_duration1 * 1000)
print("Total iterations 1:", itr_total1)

#Test 2 - do not reuse previous mean
covmeans = {}

time_start = perf_counter()
itr_total2 = 0
for p in power_list:
    mean,itr = calculate_mean(X, y, p, sample_weight, False , covmeans)
    print(itr)
    itr_total2 = itr_total2 + itr
time_end = perf_counter()
time_duration2 = time_end - time_start
print("Duration 2:", time_duration2 * 1000)
print("Total iterations 2:", itr_total2)

print("Time difference in %:", round(100 - time_duration2/time_duration1 * 100,2),"%")
print("Iteration difference in %:", round(100 - itr_total2/itr_total1 * 100,2),"%")