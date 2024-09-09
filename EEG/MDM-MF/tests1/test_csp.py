# -*- coding: utf-8 -*-
"""

====================================================================
Classification of P300 and MI datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131


Evaluation in %:
                     score      time
pipeline                            
DM_csp_or_th2     0.751579  0.854697
DM_no_csp_or_th2  0.756808  1.971264
TSLR              0.748151  0.170189

CSP enabled helps with speed.

@author: anton andreev
"""
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

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
from  enchanced_mdm_mf_tools import CustomCspTransformer

#start configuration
hb_max_n_subjects = -1
hb_n_jobs = -1
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
    
    pipeline_folder = "/home/antona/ML_examples/EEG/MDM-MF/pipelines/"
else:
    pipeline_folder = "C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines\\"

pipelines = {}
params_grid = None

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

power_means2 = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

power_means3 = [-0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

power_means4 = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1]

power_means5 = [-1, 0, 1]

power_means6 = [-0.7, -0.01, 0.4, 0.8 ]

power_means7 = [-1, 0.5, 0, 0.5, 1] #best so far

power_means8 = [-1, 0.5, 0.001, 0.5, 1]

power_means9 = [-1, -0.75, 0.5, 0.001, 0.5, 0.75, 1]

power_means10 = [-1, -0.75, -0.5, -0.3, 0.001, 0.3, 0.5, 0.75, 1]

#power_means11 = [-1, -0.75, -0.5, -0.25, -0.1, 0.001, 0.1, 0.25, 0.5, 0.75, 1]

power_means11 = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]

power_means12 = [-1, -0.75, -0.5, -0.25, -0.1, 0.001, 0.1, 0.25, 0.5, 0.75, 1]

# power_means9 = [-1]

# power_means10 = [1]

# power_means11 = [0]

#with our csp and outlier removal
pipelines["DM_csp_or_th2"] = make_pipeline(
    Covariances("oas"),
    CustomCspTransformer(nfilter = 10),
    MeanFieldNew(power_list=power_means12,
              method_label="lda",
              n_jobs=mdm_mf_jobs,
              euclidean_mean         = False, #default = false
              distance_strategy      = "default_metric",
              remove_outliers        = True,
              outliers_th            = 2.5,  #default = 2.5
              outliers_depth         = 2,    #default = 4
              max_outliers_remove_th = 50,   #default = 50
              outliers_disable_mean  = False, #default = false
              outliers_method        ="zscore",
              zeta                   = 1e-06,
              or_mean_init           = True,
              ),   
)

#no csp and outlier removal
pipelines["DM_no_csp_or_th2"] = make_pipeline(
    Covariances("oas"),
    MeanFieldNew(power_list=power_means12,
              method_label="lda",
              n_jobs=mdm_mf_jobs,
              euclidean_mean         = False, #default = false
              distance_strategy      = "default_metric",
              remove_outliers        = True,
              outliers_th            = 2.5,  #default = 2.5
              outliers_depth         = 2,    #default = 4
              max_outliers_remove_th = 50,   #default = 50
              outliers_disable_mean  = False, #default = false
              outliers_method        ="zscore",
              zeta                   = 1e-06,
              or_mean_init           = True,
              ),   
)

#with csp and no or
# pipelines["DM_csp_no_or"] = make_pipeline(
#     Covariances("oas"),
#     CustomCspTransformer(nfilter = 10),
#     MeanFieldNew(power_list=power_means12,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean         = False, #default = false
#               distance_strategy      = "default_metric",
#               remove_outliers        = False,
#               outliers_th            = 2.5,  #default = 2.5
#               outliers_depth         = 4,    #default = 4
#               max_outliers_remove_th = 50,   #default = 50
#               outliers_disable_mean  = False, #default = false
#               outliers_method        ="zscore",
#               zeta                   = 1e-06,
#               or_mean_init           = True,
#               ),   
# )

#no csp and no or
# pipelines["DM_no_csp_no_or"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer(nfilter = 10),
#     MeanFieldNew(power_list=power_means12,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean         = False, #default = false
#               distance_strategy      = "default_metric",
#               remove_outliers        = False,
#               outliers_th            = 2.5,  #default = 2.5
#               outliers_depth         = 4,    #default = 4
#               max_outliers_remove_th = 50,   #default = 50
#               outliers_disable_mean  = False, #default = false
#               outliers_method        ="zscore",
#               zeta                   = 1e-06,
#               or_mean_init           = True,
#               ),   
# )

#default csp
# from pyriemann.spatialfilters import CSP
# pipelines["DM_def_csp_or_th4"] = make_pipeline(
#     Covariances("oas"),
#     CSP(log=False),
#     MeanFieldNew(power_list=power_means12,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean         = False, #default = false
#               distance_strategy      = "default_metric",
#               remove_outliers        = True,
#               outliers_th            = 2.5,  #default = 2.5
#               outliers_depth         = 4,    #default = 4
#               max_outliers_remove_th = 50,   #default = 50
#               outliers_disable_mean  = False, #default = false
#               outliers_method        ="zscore",
#               zeta                   = 1e-06,
#               or_mean_init           = True,
#               ),   
# )

#inf means a bit better than sum means (just a little)

# pipelines["MF_orig_csp_im"] = make_pipeline(
#     Covariances("oas"),
#     CustomCspTransformer(nfilter = 10),
#     MeanField_orig(power_list=power_means,
#                     method_label="inf_means"
#               ),
# )

# pipelines["MF_orig_no_csp_im"] = make_pipeline(
#     Covariances("oas"),
#     MeanField_orig(power_list=power_means,
#                     method_label="inf_means"
#               ),
# )

#use riemann distance for the power means and the custom function

#can not use both
AUG_Tang_SVM_standard       = False #Zhou2016 subject 4 can fail because of cov covariance estimator
AUG_Tang_SVM_grid_search    = False #Zhou2016 subject 4 can fail because of cov covariance estimator
TSLR = True

from moabb.pipelines.utils import parse_pipelines_from_directory, generate_param_grid
pipeline_configs = parse_pipelines_from_directory(pipeline_folder)

if AUG_Tang_SVM_standard:
    #no Grid search
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_SVM_F"] = c["pipeline"]
            params_grid = None
            break
      
if AUG_Tang_SVM_grid_search:
    for c in pipeline_configs:
        if c["name"] == "AUG Tang SVM Grid":
            pipelines["AD_TS_GS_SVM_F"] = c["pipeline"]
            params_grid = generate_param_grid(pipeline_configs)
            break

if TSLR:
    for c in pipeline_configs:
        if c["name"] == "Tangent Space LR":
            pipelines["TSLR"] = c["pipeline"]
            break

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
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe_test.csv"
)
results.to_csv(save_path, index=True)

print("Building statistic plots")
plot_stat(results)

#plot_stat(results, removeMI = True)