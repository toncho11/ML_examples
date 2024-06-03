# -*- coding: utf-8 -*-
"""

====================================================================
Classification of MI datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131
    
The MFM-MF has these options:
    - LE - LogEuclidian mean added in additional to all power means
    - CD - custom distances 
        if p == 1:
            metric = "euclid"
        
        if p == -1:
            metric="harmonic"
        
        if p<=0.1 and p>=-0.1:
            metric = "riemann"

Pipelines:
    - ORIG
    - LDA_ORIG  (original with LDA)
    - L1_CD     (new MDM_MF with Logistic Regression with L1)
    - LDA_CD    (new MDM_MF with LDA and Custom Distances)
    - LDA_EM_CD (new MDM_MF with LDA, Custom Distances and Log Euclidean Mean) 
    
Results:
    
    This is only MI within session.
    
    Evaluation in %:
                  score      time
    pipeline                     
    LDA_CD     0.771369  2.139092
    LDA_EM     0.748174  4.645749
    LDA_EM_CD  0.761737  2.163243
    LDA_ORIG   0.753366  8.228932
    ORIG       0.715032  2.033205
    
    The results show that:
        - using both LDA and CD provide the best result: 5.6% more than ORIG.
            - LDA adds 3.8% (2/3) roughly
            - and CD adds approximately 1.8% more (1/3)
        - EM 
            - decreases with 1% when used with combination with LDA and CD
            - still it is slightly better in terms of SMD, but result is not significant at all
            
    Note: it is currently not possible to run ORIG + CD only (but if needed could be)
    
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
hb_overwrite = False #if you change the MDM_MF algorithm you need to se to True
mdm_mf_jobs = 1
#end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}
params_grid = None

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

pipelines["ORIG"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField_orig(power_list=power_means,
              method_label="inf_means",
              n_jobs=mdm_mf_jobs,
              ),
)

#no LDA, just custom distances
#not possible currently
# pipelines["CD_EM_ORIG"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanFieldNew(power_list=power_means,
#               method_label="inf_means", #not used if used as transformer
#               n_jobs=mdm_mf_jobs,
#               euclidean_mean=False,
#               custom_distance=True
#               )
# )

pipelines["LDA_ORIG"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanFieldNew(power_list=power_means,
              method_label="inf_means", #not used if used as transformer
              n_jobs=mdm_mf_jobs,
              euclidean_mean=False,
              custom_distance=False
              ),
    LDA()
)

pipelines["LDA_EM"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanFieldNew(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=mdm_mf_jobs,
              euclidean_mean=True,
              custom_distance=False
              ),
    LDA()
)

pipelines["LDA_CD"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanFieldNew(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=mdm_mf_jobs,
              euclidean_mean=False,
              custom_distance=True
              ),
    LDA()
)

pipelines["LDA_EM_CD"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanFieldNew(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=mdm_mf_jobs,
              euclidean_mean=True,
              custom_distance=True
              ),
    LDA()
)

results = benchmark_alpha(pipelines, 
                          #params_grid = params_grid, 
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

#plot_stat(results, removeMI_LR = True)