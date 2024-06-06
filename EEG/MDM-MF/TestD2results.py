# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:49:39 2024

@author: antona
"""

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)

from pyriemann.classification import MDM
import os
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
from moabb import set_log_level

# P300 databases
from moabb.datasets import (
    BI2013a,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_003,
    EPFLP300,
    Lee2019_ERP,
    BI2014a,
    BI2014b,
    BI2015a,
    BI2015b,
)

# Motor imagery databases
from moabb.datasets import (
    BNCI2014_001,
    Zhou2016,
    BNCI2015_001,
    BNCI2014_002,
    BNCI2014_004,
    #BNCI2015_004, #not tested
    AlexMI,
    Weibo2014,
    Cho2017,
    GrosseWentrup2009,
    PhysionetMI,
    Shin2017A,
    Lee2019_MI, #new
    Schirrmeister2017 #new
)
from moabb.paradigms import P300, MotorImagery, LeftRightImagery

datasets_LR = [
    #BNCI2014_001(), #D2
    #BNCI2014_004(), #D2
    Cho2017(),      #D2
    #GrosseWentrup2009(), #D2
    #PhysionetMI(), #D2
    #Shin2017A(accept=True), #D2
    #Weibo2014(), #D2
    
    #Zhou2016(), #D2 #gives error on cov matrix not PD D2
    #new datasets
    #Lee2019_MI(), #D2 #not downloadable
    #Schirrmeister2017() #D2
]

paradigm_LR = LeftRightImagery()
AUG_Tang_SVM_standard    = False #no grid search
AUG_Tang_SVM_grid_search = True

from moabb.pipelines.utils import parse_pipelines_from_directory, generate_param_grid
pipeline_configs = parse_pipelines_from_directory("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines\\")

pipelines = {}

if AUG_Tang_SVM_standard:
    #no GS
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_SVM_F"] = c["pipeline"]
    params_grid = None
      
if AUG_Tang_SVM_grid_search:
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_GS_SVM_F"] = c["pipeline"]
    params_grid = generate_param_grid(pipeline_configs)
    
evaluation_LR = WithinSessionEvaluation(
    paradigm=paradigm_LR,
    datasets=datasets_LR,
    suffix="examples",
    overwrite=True,
    n_jobs=24,
    n_jobs_evaluation=24,
    #cache_config=cache_config,
)

results_LR = evaluation_LR.process(pipelines, param_grid=params_grid)

from heavy_benchmark import plot_stat
plot_stat(results_LR)