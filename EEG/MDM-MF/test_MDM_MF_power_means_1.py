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
hb_max_n_subjects = -1
hb_n_jobs = -1
hb_overwrite = False #if you change the MDM_MF algorithm you need to se to True
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

#labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}
params_grid = None

#csp_filters = 8

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

power_means2 = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

power_means3 = [-0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

power_means4 = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1]

power_means5 = [-1, 0, 1]

power_means6 = [-0.7, -0.01, 0.4, 0.8 ]

power_means7 = [-1, 0.5, 0, 0.5, 1] #best so far

power_means8 = [-1, 0.5, 0.001, 0.5, 1]

# power_means9 = [-1]

# power_means10 = [1]

# power_means11 = [0]

from pyriemann.spatialfilters import CSP

# pipelines["PM2_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means2,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),
# )

# pipelines["PM3_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means3,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),
# )

# pipelines["PM4_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means4,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),
# )

# pipelines["PM5_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means5,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),
    
# )


# pipelines["PM6_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means6,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),
# )

# pipelines["PM7_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means7,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),   
# )

# pipelines["PM8_LDA_CD"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means8,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =False,
#               outliers_th = 2.5
#               ),   
# )

pipelines["PM8_LDA_CD_RO_TH_2.5"] = make_pipeline(
    Covariances("oas"),
    MeanFieldNew(power_list=power_means8,
              method_label="lda",
              n_jobs=mdm_mf_jobs,
              euclidean_mean  =False,
              custom_distance =True,
              remove_outliers =True,
              outliers_th = 2.5
              ),   
)

pipelines["CSP_PM8_LDA_CD_RO"] = make_pipeline(
    Covariances("oas"),
    CSP(log=False, nfilter=8, metric="ale"), #manually set to 120 max iterations
    MeanFieldNew(power_list=power_means8,
              method_label="lda",
              n_jobs=mdm_mf_jobs,
              euclidean_mean  =False,
              custom_distance =True,
              remove_outliers =True,
              outliers_th = 2.5
              ),   
)



# pipelines["PM_LDA_CD_RO_TH_2.5"] = make_pipeline(
#     Covariances("oas"),
#     MeanFieldNew(power_list=power_means,
#               method_label="lda",
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean  =False,
#               custom_distance =True,
#               remove_outliers =True,
#               outliers_th = 2.5
#               ),   
# )

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