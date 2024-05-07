# -*- coding: utf-8 -*-
"""

====================================================================
Classification of P300 datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Testing if more power means will improve the results.
It uses several methods for feature selection (to reduce the number of means used):
    - SFM - Select From Model (ex. LinearSVC)
    - SFS - Sequential Feature Selector (ex. LogisticRegression)

Tests several algorithms running after MFM_MF:
    - LR - logistic regression
    - LR_L1 - logistic regression L1
    - LDA

The MFM-MF has these options:
    - LE - LogEuclidian mean added in additional to all power means
    - CD - custom distances 
        - the distance to the LogEuclidian mean is LogEuclidian
        - the disance to power mean p=1 is Euclidian

Results:

    
    
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

#start configuration
hb_max_n_subjects = 10
hb_n_jobs = 24
hb_overwrite = True #if you change the MDM_MF algorithm you need to se to True
#end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

power_means_extended = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

power_means_extended_LEM = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200]

#both : log euclidian with riemann distance and logeuclidian with log euclidian distance
power_means_extended_LEM_LED = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200, 300]

power_means_LEM_LED = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 200, 300]

# power_means_extended2 = [-1, -0.9, -0.1, 0.1, 0.9, 1]

# power_means_extended3 = [-1, -0.9, -0.01, 0.01, 0.9, 1]

# power_means_extended4 = [0.01, 0.9, 1]

# power_means_extended5 = [-1, -0.9, -0.01]

pipelines["MF_orig"] = make_pipeline(
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
              n_jobs=12,
              
              ),
)

pipelines["LDA"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    LDA()
)

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
pipelines["LDA_SFM_LE"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means_extended_LEM_LED,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    SelectFromModel(LinearSVC(dual="auto", penalty="l1")),
    LDA()
)

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
pipelines["LDA_SFM_FO_LE"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means_extended_LEM_LED,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    SelectFromModel(RandomForestClassifier(), threshold='median'),
    LDA()
)

pipelines["L1_SFM_LE"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means_extended_LEM_LED,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    SelectFromModel(LinearSVC(dual="auto", penalty="l1")),
    LogisticRegression(penalty="l1", solver="liblinear")
)

# from mlxtend.feature_selection import SequentialFeatureSelector
# pipelines["LDA_SFS_LR_LE"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField(power_list=power_means_extended_LE,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=12,
#               ),
#     SequentialFeatureSelector(estimator = LogisticRegression()),
#     LDA()
# )

# from mlxtend.feature_selection import SequentialFeatureSelector
# from sklearn.neighbors import KNeighborsClassifier
# pipelines["LDA_SFS_KNN_LE"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField(power_list=power_means_extended_LE,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=12,
#               ),
#     SequentialFeatureSelector(estimator =  KNeighborsClassifier(n_neighbors=5)),
#     LDA()
# )



results = benchmark_alpha(pipelines, 
                          evaluation_type="withinsession", 
                          max_n_subjects = hb_max_n_subjects, 
                          n_jobs=hb_n_jobs, 
                          overwrite = hb_overwrite,
                          #skip_MR_LR=True
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