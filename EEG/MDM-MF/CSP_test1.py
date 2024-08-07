# -*- coding: utf-8 -*-
"""

====================================================================
Classification of P300 and MI datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Pipelines:
    - ORIG
    - L1_CD  (new MDM_MF with Logistic Regression with L1)
    - LDA_CD (new MDM_MF with LDA)

The MFM-MF has these options:
    - LE - LogEuclidian mean added in additional to all power means
    - CD - custom distances 
        if p == 1:
            metric = "euclid"
        
        if p == -1:
            metric="harmonic"
        
        if p<=0.1 and p>=-0.1:
            metric = "riemann"

Results:
    
   
            
@author: anton andreev
"""

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from enchanced_mdm_mf import MeanField as MeanFieldNew
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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#start configuration
hb_max_n_subjects = 1
hb_n_jobs = 12
hb_overwrite = True #if you change the MDM_MF algorithm you need to se to True
mdm_mf_jobs = 1
is_on_grid = False
#end configuration

if is_on_grid:
    from mne import get_config, set_config
    print("Changing MNE folder ...")
    set_config('MNE_DATA', '/silenus/PROJECTS/pr-eeg-dl/antona/')
    new_path = get_config("MNE_DATA")
    print(f"The download directory is currently {new_path}")
    print("Done changing MNE folder")

#labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}
params_grid = None

#csp_filters = 8

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

from pyriemann.spatialfilters import CSP

pipelines["LDA_CD"] = make_pipeline(
    Covariances("oas"),
    MeanFieldNew(power_list=power_means,
              n_jobs=mdm_mf_jobs,
              euclidean_mean=False,
              custom_distance=True
              ),
    LDA()
)

# from sklearn.model_selection import GridSearchCV
# param_grid_csp =  {
#                     'nfilter': [4,8,10],
#                     'log'    : [False],
#                     #'metric' : ["ale", "alm", "euclid", "harmonic", "identity", "kullback_sym", "logdet", "logeuclid", "riemann", "wasserstein",]
#                     'metric' : ["harmonic", "logdet", "logeuclid", "riemann", "euclid"]
#                   }

# grid_csp =  GridSearchCV(CSP(), param_grid_csp , refit = True, verbose = 0, scoring='balanced_accuracy') 

# param_grid_mf =  {
#                     'euclidean_mean' : [True, False],
#                     'custom_distance': [True, False]
#                   }

# grid_mf =  GridSearchCV(MeanFieldNew(power_list=power_means), param_grid_mf , refit = True, verbose = 0, scoring='accuracy') 

# pipelines["CSP_GS_M_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     grid_csp, #CSP has none of the following attributes: predict.
#     MeanFieldNew(power_list=power_means,
#                   n_jobs=mdm_mf_jobs,
#                   euclidean_mean=False,
#                   custom_distance=True
#                   ),
#     LDA()
# )

# pipelines["CSP_GS_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     grid_csp,
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_8_LDA_CD_H"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=8,metric="harmonic"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =True,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_8_LDA_CD_alm"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=8,metric="alm"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =True,
#               custom_distance=True
#               ),
#     LDA()
# )

# current best !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# cov CSP_8_LDA_CD_ale
# pipelines["CSP_8_LDA_CD_ale"] = make_pipeline(
#     Covariances("cov"),
#     CSP(log=False, nfilter=8,metric="ale"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_8_LDA_EU_CD_ale_cov"] = make_pipeline(
#     Covariances("cov"),
#     CSP(log=False, nfilter=8, metric="ale"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =True,
#               custom_distance=True
#               ),
#     LDA()
# )

#INFO convegence increased to 200 in ajd.py
pipelines["CSP_8_LDA_EU_CD_ale_oas"] = make_pipeline(
    Covariances("oas"),
    CSP(log=False, nfilter=8, metric="ale", ),
    MeanFieldNew(power_list=power_means,
              n_jobs=mdm_mf_jobs,
              euclidean_mean =False,
              custom_distance=True
              ),
    LDA()
)

pipelines["CSP_8_LDA_CD_ale_oas"] = make_pipeline(
    Covariances("oas"),
    CSP(log=False, nfilter=8, metric="ale", ),
    MeanFieldNew(power_list=power_means,
              n_jobs=mdm_mf_jobs,
              euclidean_mean =False,
              custom_distance=True
              ),
    LDA()
)

pipelines["CSP_LDA_CD_oas"] = make_pipeline(
    Covariances("oas"),
    CSP(log=False),
    MeanFieldNew(power_list=power_means,
              n_jobs=mdm_mf_jobs,
              euclidean_mean =False,
              custom_distance=True
              ),
    LDA()
)


# pipelines["CSP_8_LDA_CD_euc"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=8,metric="euclid"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_6_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=6),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_4_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=4),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["CSP_12_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter=12),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LDA()
# )

# pipelines["QDA"] = make_pipeline(
#     Covariances("cov"),
#     MeanFieldNew(power_list=power_means,
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean =False,
#               custom_distance=True
#               ),
#     QDA()
# )

# pipelines["MDM"] = make_pipeline(
#     Covariances("oas"),
#     MDM()
# )

# pipelines["CSP_MDM"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False, nfilter = csp_filters),
#     MDM()
# )

#can not use both
AUG_Tang_SVM_standard       = False #Zhou2016 subject 4 can fail because of cov covariance estimator
AUG_Tang_SVM_grid_search    = False #Zhou2016 subject 4 can fail because of cov covariance estimator
TSLR = True

from moabb.pipelines.utils import parse_pipelines_from_directory, generate_param_grid
pipeline_configs = parse_pipelines_from_directory("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines\\")

if AUG_Tang_SVM_standard:
    #no Grid search
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_SVM_F"] = c["pipeline"]
    params_grid = None
      
if AUG_Tang_SVM_grid_search:
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_GS_SVM_F"] = c["pipeline"]
    params_grid = generate_param_grid(pipeline_configs)

if TSLR:
    for c in pipeline_configs:
        if c["name"] == "Tangent Space LR":
            pipelines["TSLR"] = c["pipeline"]

results = benchmark_alpha(pipelines, 
                          params_grid = params_grid, 
                          evaluation_type="withinsession",
                          #evaluation_type="crosssubject", 
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

#plot_stat(results, removeMI = True)