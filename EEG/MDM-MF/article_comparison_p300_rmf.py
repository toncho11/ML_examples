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
from enchanced_mdm_mf_tools import CustomCspTransformer
from pyriemann.spatialfilters import CSP
from moabb import benchmark, set_log_level

#start configuration
# hb_max_n_subjects = 2
# hb_n_jobs = -1
# hb_overwrite = True #if you change the MDM_MF algorithm you need to se to True
# mdm_mf_jobs = 1
is_on_grid = False
#end configuration

if is_on_grid:
    from mne import get_config, set_config
    print("Changing MNE folder ...")
    set_config('MNE_DATA', '/silenus/PROJECTS/pr-eeg-dl/antona/')
    new_path = get_config("MNE_DATA")
    print(f"The download directory is currently {new_path}")
    print("Done changing MNE folder")
    
    pipeline_folder = "/home/antona/ML_examples/EEG/MDM-MF/pipelines5/"
else:
    pipeline_folder = "C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines5\\"

#The EN_grid.yml is too slow!!!!!!!! and memory consuming

results = benchmark(
    pipelines=pipeline_folder,
    evaluations=["WithinSession"],
    paradigms=["P300"],
    #include_datasets=["Zhou2016"],
    #exclude_datasets=["Stieger2021"],
    results="./results/",
    overwrite=False,
    plot=True,
    n_jobs=4, #otherwise memory is not enough
    output="./benchmark/",
)

print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

# save results
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe_test_moabb.csv"
)
results.to_csv(save_path, index=True)

print("Building statistic plots")
plot_stat(results)

#plot_stat(results, removeMI = True)