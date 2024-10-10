# -*- coding: utf-8 -*-
"""

====================================================================
Classification of P300 and MI datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Results:
    
    Evaluation in %:
                         score      time
    pipeline                            
    MDM               0.634990  0.177995
    MF_orig           0.656966  1.996332
    MF_orig_cspa      0.686152  0.279583
    TSLR              0.750644  0.187834
    new_best          0.756664  0.270816
    new_best_csp_def  0.721146  0.169360
    new_best_no_csp   0.751160  2.749826
    new_best_no_or    0.748291  0.268579
    
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
from enchanced_mdm_mf_tools import CustomCspTransformer2
from pyriemann.spatialfilters import CSP

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

pipelines["MDM"] = make_pipeline(
    Covariances("oas"),
    MDM(),
)

pipelines["MDM_OR"] = make_pipeline(
    Covariances("oas"),
    MDM(metric="riemann_or"),
)

from pyriemann.tangentspace import TangentSpace

pipelines["TS"] = make_pipeline(
    Covariances("oas"),
    TangentSpace(),
    LogisticRegression(C=1.0)
)

pipelines["TS_OR"] = make_pipeline(
    Covariances("oas"),
    TangentSpace(metric="riemann_or"),
    LogisticRegression(C=1.0)
)

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