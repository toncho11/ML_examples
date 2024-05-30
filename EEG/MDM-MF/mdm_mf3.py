# -*- coding: utf-8 -*-
"""

====================================================================
Classification of MI datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Testing if more power means will improve the results.
It uses several methods for feature selection (to reduce the number of means used):
    - SFM - Select From Model (ex. LinearSVC)
    - PL  - PolynomialFeatures (actually adds more features)

When p=200 an Euclidean mean is created with Riemann distance
When p=300 an Euclidean mean is created with Eucledan distance

Tests several algorithms running after MFM_MF:
    
    - MF_orig - original MFM when published
      - PM: [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
      
    - LDA_SFM_LE
      - Uses more power means, Euclidean mean with both Riemann distance and Euclidean distance (200,300)
          - PM + EU: [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200, 300]
      - Uses Select From Model for feature selection with estimator LinearSVC (l1)
      - LDA
      
    - PL_LDA 
        - Uses more power means, Euclidean mean with both Riemann distance and Euclidean distance (200,300)
            - PM + EU: [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200, 300]
        - Uses PolynomialFeatures preprocessor before classifications 
        - LDA
        
    - PL_SFO_LDA 
        - Uses more power means, Euclidean mean with both Riemann distance and Euclidean distance (200,300)
            - PM + EU: [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200, 300]
        - Uses PolynomialFeatures preprocessor (degree = 2, bias added = true)
        - Uses Select From Model for feature selection with estimator LinearSVC (l1)
        - LDA
        
    - LDA_LE
        - Uses more power means, Euclidean mean with both Riemann distance and Euclidean distance (200,300)
            - PM + EU: [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200, 300]
        - Logistic Regression l1 as final classifier
        
    - LDA
        - PM: [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
        - as MF_orig but with LDA
    

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

class SelectFromModelEx(SelectFromModel):

    def _get_feature_importances(self,estimator):
        """Retrieve or aggregate feature importances from estimator"""
        importances = getattr(estimator, "feature_importances_", None)
        
        if importances is None and hasattr(estimator, "coef_"):
            if estimator.coef_.ndim == 1:
                importances = np.abs(estimator.coef_)
        
            else:
                importances = np.sum(np.abs(estimator.coef_), axis=0)
        
        elif importances is None:
            raise ValueError(
                "The underlying estimator %s has no `coef_` or "
                "`feature_importances_` attribute. Either pass a fitted estimator"
                " to SelectFromModel or call fit before calling transform."
                % estimator.__class__.__name__)
    
        return importances

    def fit(self, X, y=None, **fit_params):
        print("Fitting SelectFromModelEx model estimator ...")
        super().fit(X, y, **fit_params)
        importances = self._get_feature_importances(self)
        print("Done fitting SelectFromModelEx model estimator ...")
        
        return self

#start configuration
hb_max_n_subjects = 10
hb_n_jobs = 24
hb_overwrite = True #if you change the MDM_MF algorithm you need to se to True
mdm_mf_jobs = 1
#end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}
params_grid = None

power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

#power_means_extended = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

#power_means_extended_LEM = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200]

#both : log euclidian with riemann distance and logeuclidian with log euclidian distance
#power_means_extended_LEM_LED = [-1, -0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 200]

#power_means_LEM_LED = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 200]

# power_means_extended2 = [-1, -0.9, -0.1, 0.1, 0.9, 1]

# power_means_extended3 = [-1, -0.9, -0.01, 0.01, 0.9, 1]

# power_means_extended4 = [0.01, 0.9, 1]

# power_means_extended5 = [-1, -0.9, -0.01]

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

#########################################################################################
# pipelines["LDA"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField(power_list=power_means_LEM_LED,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=mdm_mf_jobs,
#               ),
#     #LogisticRegression(penalty="l1", solver="liblinear")
#     LDA()
# )

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