# -*- coding: utf-8 -*-
"""

====================================================================
Classification of P300 datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Tests several algorithms running after MFM_MF:
    - LR - logistic regression
    - LR_L1 - logistic regression L1
    - LDA
    - PCA

The MFM-MF has these options:
    - LE - LogEuclidian mean added in additional to all power means
    - CD - custom distances 
        - the distance to the LogEuclidian mean is LogEuclidian
        - the disance to power mean p=1 is Euclidian

Results:
    - LDA seems always the best in general and Motor Imagery
        - LE_CD_PCA_LDA seems to be close and statistically better than LDA for some datasets
    - LE_CD_PCA_LR_L1 is the best for P300
        - better than LDA
        - better than non PCA version
        - better than LE_PCA_LDA
    
    Adding PCA seems to help (components = 6).
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

#start configuration
hb_max_n_subjects = 10
hb_n_jobs = 24
hb_overwrite = True #if you change the MDM_MF algorithm you need to se to True
#end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}

power_means = [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

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

#for 200 the mean_logeuclid is calculated instead of power mean
power_means = [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 200]

from sklearn.decomposition import PCA
pipelines["LE_PCA_LDA"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              custom_distance = False,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    PCA(n_components=6),#n_components=7
    LDA()
)

pipelines["LE_CD_PCA_LDA"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              custom_distance = True,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    PCA(n_components=6),#n_components=7
    LDA()
)

pipelines["LE_CD_LR_L1"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              custom_distance = True,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    LogisticRegression(penalty="l1", solver="saga", n_jobs=6)
    #LogisticRegression(penalty="l1", solver="liblinear")
)

pipelines["LE_CD_PCA_LR_L1"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              custom_distance = True,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    PCA(n_components=6),#n_components=7
    LogisticRegression(penalty="l1", solver="liblinear", n_jobs=6)
    #LogisticRegression(penalty="l1", solver="liblinear")
)

results = benchmark_alpha(pipelines, max_n_subjects = hb_max_n_subjects, n_jobs=hb_n_jobs, overwrite = hb_overwrite)

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