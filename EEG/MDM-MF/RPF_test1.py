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
    
    This is both P300 and MI withinsession.
    
    1) Overall result below shows:
    
    Evaluation in %:
                 score      time
    pipeline                    
    L1_CD     0.842027  1.166564
    LDA_CD    0.855700  1.144555
    ORIG      0.834115  1.023850
    
    The overall result is with 2.1% better than ORIG due to the gain in Motor Imagery, but not P300.
    
    
    2) Only Motor Imagery is shown below it shows that both CD and LDA
    contribute to a 6.6% percent difference with ORIG and LDA_CD is signficantly better
    than the other 2: ORIG and L1_CD
        
    Evaluation in %:
                 score      time
    pipeline                    
    L1_CD     0.710396  2.299669
    LDA_CD    0.758411  2.281411
    ORIG      0.692389  2.157802
    
    3) On P300 there is no difference in terms of ROC AUC average performance:
        
    Evaluation in %:
                 score      time
    pipeline                    
    L1_CD     0.903280  0.586287
    LDA_CD    0.900043  0.562840
    ORIG      0.900399  0.445507
    
    but L1_CD is signficantly better than ORIG and LDA_CD with 3 dots.
    So this confirms that LR_L1 is always better than LDA for P300.
            
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
from pyriemann.clustering import Potato, PotatoField

#start configuration
hb_max_n_subjects = 3
hb_n_jobs = 1
hb_overwrite = False #if you change the MDM_MF algorithm you need to se to True
mdm_mf_jobs = 1
#end configuration

labels_dict = {"Target": 1, "NonTarget": 0}
pipelines = {}
params_grid = None

z_th = 2.0           # z-score threshold
low_freq, high_freq = 1., 35.
p_th = 0.01          # probability threshold
rpf_config = {
    'RPF eye_blinks': {  # for eye-blinks
        'ch_names': ['Fp1', 'Fpz', 'Fp2'],
        'low_freq': 1.,
        'high_freq': 20.},
    'RPF occipital': {  # for high-frequency artifacts in occipital area
        'ch_names': ['O1', 'Oz', 'O2'],
        'low_freq': 25.,
        'high_freq': 45.,
        'cov_normalization': 'trace'},  # trace-norm to be insensitive to power
    'RPF global_lf': {  # for low-frequency artifacts in all channels
        'ch_names': None,
        'low_freq': 0.5,
        'high_freq': 3.}
}
rpf = PotatoField(metric='riemann', z_threshold=z_th, p_threshold=p_th,
                  n_potatoes=len(rpf_config))

#https://github.com/pyRiemann/pyRiemann/blob/7f599a93202d9636e32e590cd7d0bbf831de2b6d/examples/artifacts/plot_detect_riemannian_potato_field_EEG.py#L25
pipelines["MDM"] = make_pipeline(
    #there should be another estimator that
           #applies the above filters
           #produces Cov matrices
           #shape 4 (n_potators, n_matrices, n_channels, n_channels)
    rpf,  #it should have another estimator that
    MDM(),
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

#plot_stat(results, removeMI = True)