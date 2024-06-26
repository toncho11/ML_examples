# -*- coding: utf-8 -*-
"""

====================================================================
Classification of only MI datasets from MOABB using MDM-MF
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
    
                       score      time
    pipeline                          
    AD_TS_GS_SVM_F  0.722002      0.688727
    LDA_CD          0.782034      5.999502
    LDA_CD_SCM      0.763893      5.568990
    LR_CV_CD        0.768435      5.544800
    TSLR            0.808061      0.471176
    MDM             0.641112      0.239932
    MF_orig         0.654631      2.919833
    
            
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#start configuration
hb_max_n_subjects = -1
hb_n_jobs = 24
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

pipelines = {}
params_grid = None

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

pipelines["LDA_CD"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    Covariances("oas"),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanFieldNew(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=mdm_mf_jobs,
              euclidean_mean=False,
              custom_distance=True
              ),
    LDA()
)

# pipelines["LDA_CD_SCM"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     Covariances("scm"), #instead of oas
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanFieldNew(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LDA()
# )

# from sklearn.model_selection import GridSearchCV 
# from sklearn.linear_model import Lasso

# param_grid_lasso = {'alpha': (np.logspace(-8, 8, 100))}
# grid_lasso = GridSearchCV(Lasso(), param_grid_lasso, refit = True, verbose = 0, cv = 5) 

# pipelines["LASSO_CV_CD"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     Covariances("oas"),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanFieldNew(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     Lasso(alpha=0.1)
# )

# from sklearn.linear_model import LogisticRegressionCV
# pipelines["LR_CV_CD"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     Covariances("oas"),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanFieldNew(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               ),
#     LogisticRegressionCV()
# )

#previous algorithms to compare with
pipelines["MF_orig"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    Covariances("oas"),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField_orig(power_list=power_means,
              method_label="inf_means",
              n_jobs=mdm_mf_jobs,
              
              ),
)

pipelines["MDM"] = make_pipeline(
    Covariances("oas"),
    MDM(),
)

#STA pipelines to compare with

#can not use both
AUG_Tang_SVM_standard       = False #Zhou2016 subject 4 can fail because of cov covariance estimator
AUG_Tang_SVM_grid_search    = False #Zhou2016 subject 4 can fail because of cov covariance estimator
TSLR = False

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