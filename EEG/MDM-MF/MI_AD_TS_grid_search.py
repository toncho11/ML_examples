# -*- coding: utf-8 -*-
"""

====================================================================
Searching for the best pipeline for Motor imagery
====================================================================

Pipelines:
    -
    -
    -
    
Results 1:
    In most cases 8/10 cases AD_TS_GR_SVM and TS_GR_SVM are better.
    AD_TS_GR_SVM TS_GR_SVM seem to give identical results.
    
    AD_TS_GR_SVM  0.721195  97.114830
    TS_GR_SVM     0.721197  29.016794
    TS_LDA        0.693394  20.043756
    estimator='cov'

    
    
@author: anton andreev
"""

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from enchanced_mdm_mf import MeanField
from pyriemann.classification import MeanField as MeanField_orig
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)

from pyriemann.classification import MDM
import os
from heavy_benchmark import benchmark_alpha, plot_stat
from moabb import set_download_dir
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from pyriemann.tangentspace import TangentSpace
from moabb.pipelines.features import AugmentedDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 

#start configuration
hb_max_n_subjects = 10 #2 subjects works
hb_n_jobs = 24
hb_overwrite = False #if you change the MDM_MF algorithm you need to se to True
#end configuration

pipelines = {}
params_grid  = None

pipelines["TS_LDA"] = make_pipeline(
    Covariances(),
    TangentSpace(),
    LDA()#LogisticRegression(penalty="l1", solver="liblinear")
)

# from moabb.pipelines.utils import parse_pipelines_from_directory
# pipeline_configs = parse_pipelines_from_directory("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines\\")
# for c in pipeline_configs:
#     if c["name"] == "AUG Tang SVM Grid":
#         pipelines["AD_TS_GR_SVM_F"] = c["pipeline"]

# param_grid_ad =  {
#                    'order': [1,2,3,4,5,6,7,8,9,10],  
#                      'lag': [1,2,3,4,5,6,7,8,9,10]
#                  }

# param_grid_svc = {         'C': [0.5, 1.0, 1.5],  
#                   'kernel': ['rbf','linear'],      
#                   }

# grid_ad =  GridSearchCV(AugmentedDataset(), param_grid_ad , refit = True, verbose = 0, scoring='balanced_accuracy') 
# grid_svc = GridSearchCV(SVC()             , param_grid_svc, refit = True, verbose = 0, scoring='balanced_accuracy') 

# augmnetation and grid search
# pipelines["AD_TS_GR_SVM"] = make_pipeline(
#     grid_ad,
#     Covariances(),
#     TangentSpace(),
#     grid_svc,
# ) 

# # no augmentation and no gridsearch
# pipelines["TS_SVM"] = make_pipeline(
#     Covariances(),
#     TangentSpace(),
#     SVC(),
# )

# # augmentation and no grid search
# pipelines["AD_TS_SVM"] = make_pipeline(
#     AugmentedDataset(),
#     Covariances(),
#     TangentSpace(),
#     SVC(),
# )

from moabb.pipelines.utils import parse_pipelines_from_directory, generate_param_grid
pipeline_configs = parse_pipelines_from_directory("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines\\")

params_grid = generate_param_grid(pipeline_configs)

for c in pipeline_configs:
    if c["name"] == "AUG Tang SVM Grid":
        pipelines["AD_TS_GR_SVM_F2"] = c["pipeline"]

results = benchmark_alpha(pipelines, 
                          params_grid = params_grid, 
                          #evaluation_type="withinsession",
                          evaluation_type="crosssubject", 
                          max_n_subjects = hb_max_n_subjects, 
                          n_jobs=hb_n_jobs, 
                          overwrite = hb_overwrite,
                          skip_P300 = True,
                          skip_MI   = False,
                          replace_x_dawn_cov_par_cov_for_MI=True
                          )

print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# save results
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe.csv"
)
results.to_csv(save_path, index=True)

print("Building statistic plots")
plot_stat(results)

#plot_stat(results, removeMI_LR = True)