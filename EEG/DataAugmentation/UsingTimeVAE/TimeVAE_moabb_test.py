# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:39:07 2024

@author: antona

Results:
                                               score      time
    dataset        pipeline                                   
    BNCI2014-001_M TSLR                      0.874048  0.118572
                   TimeVAE+LDA_35F_PSF       0.676266  1.742716
                   TimeVAE+LDA_35F_PST       0.708760  1.682656
                   TimeVAE+LDA_35F_PST_CSP   0.655026  1.580568
                   TimeVAE+LDA_35F_PST_LD12  0.715994  1.676706
    Zhou2016_M     TSLR                      0.946686  0.056751
                   TimeVAE+LDA_35F_PSF       0.749553  1.514838
                   TimeVAE+LDA_35F_PST       0.765359  1.518037
                   TimeVAE+LDA_35F_PST_CSP   0.656526  1.647291
                   TimeVAE+LDA_35F_PST_LD12  0.756906  1.560706
"""

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from moabb.evaluations import (
    WithinSessionEvaluation,
    CrossSessionEvaluation,
    CrossSubjectEvaluation,
)

from pyriemann.classification import MDM
import os
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
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from pyriemann.spatialfilters import CSP
from time import perf_counter
from TimeVaeTransformer import TimeVaeTransformer

import sys
#unclude benchmark
sys.path.insert(1, 'C:/Work/PythonCode/ML_examples/EEG/MDM-MF')
from heavy_benchmark import benchmark_alpha, plot_stat
from enchanced_mdm_mf_tools import CustomCspTransformer,CustomCspTransformer2, CustomCspTransformer3

#filter messages form tensorflow
#Warning: can hide important messages!!!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

#start configuration
hb_max_n_subjects = 10
hb_n_jobs = 1
hb_overwrite = False #if you change the MDM_MF algorithm you need to set to True
is_on_grid = False
#end configuration

if is_on_grid:
    from mne import get_config, set_config
    print("Changing MNE folder ...")
    set_config('MNE_DATA', '/silenus/PROJECTS/pr-eeg-dl/antona/')
    new_path = get_config("MNE_DATA")
    print(f"The download directory is currently {new_path}")
    print("Done changing MNE folder")
    
    pipeline_folder = "/home/antona/ML_examples/EEG/MDM-MF/pipelines3/"
else:
    pipeline_folder = "C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\pipelines3\\"

pipelines = {}
params_grid = None

# pipelines["TimeVAE+SVC_3"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, vae_iterations = 50),
#     svm.SVC()
# )

# pipelines["TimeVAE+LDA_35F_PSF"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, 
#                        vae_iterations = 5, 
#                        use_augmented_data = False, 
#                        use_pretrained_scaler = False),
#     LDA()
# )

#best
# pipelines["TimeVAE+LDA_35F_PST"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, 
#                        vae_iterations = 5, 
#                        use_augmented_data = False, 
#                        use_pretrained_scaler = True),
#     LDA()
# )

# pipelines["TimeVAE+LDA_35F_PST_CSP"] = make_pipeline(
#     Covariances("oas"),
#     CustomCspTransformer2(mode="high_electrodes_count"),
#     CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, 
#                        vae_iterations = 5, 
#                        use_augmented_data = False, 
#                        use_pretrained_scaler = True),
#     LDA()
# )

pipelines["TimeVAE+LDA_35F_PST_LD12"] = make_pipeline(
    Covariances("oas"),
    #CustomCspTransformer2(mode="high_electrodes_count"),
    #CustomCspTransformer2(mode="low_electrodes_count"),
    TimeVaeTransformer(encoder_output = 3, 
                       vae_iterations = 5, 
                       use_augmented_data = False, 
                       use_pretrained_scaler = True,
                       vae_latent_dim = 12,
                       ),
    LDA()
)

pipelines["TimeVAE+LDA_35F_PST_LD16"] = make_pipeline(
    Covariances("oas"),
    #CustomCspTransformer2(mode="high_electrodes_count"),
    #CustomCspTransformer2(mode="low_electrodes_count"),
    TimeVaeTransformer(encoder_output = 3, 
                       vae_iterations = 5, 
                       use_augmented_data = False, 
                       use_pretrained_scaler = True,
                       vae_latent_dim = 16,
                       ),
    LDA()
)

pipelines["TimeVAE+LDA_33F_PST_LD16"] = make_pipeline(
    Covariances("oas"),
    #CustomCspTransformer2(mode="high_electrodes_count"),
    #CustomCspTransformer2(mode="low_electrodes_count"),
    TimeVaeTransformer(encoder_output = 3, 
                       vae_iterations = 3, 
                       use_augmented_data = False, 
                       use_pretrained_scaler = True,
                       vae_latent_dim = 16,
                       ),
    LDA()
)

pipelines["TimeVAE+LDA_310F_PST_LD20"] = make_pipeline(
    Covariances("oas"),
    #CustomCspTransformer2(mode="high_electrodes_count"),
    #CustomCspTransformer2(mode="low_electrodes_count"),
    TimeVaeTransformer(encoder_output = 3, 
                       vae_iterations = 10, 
                       use_augmented_data = False, 
                       use_pretrained_scaler = True,
                       vae_latent_dim = 20,
                       ),
    LDA()
)

pipelines["TimeVAE+LDA_35F_PST_LD20"] = make_pipeline(
    Covariances("oas"),
    #CustomCspTransformer2(mode="high_electrodes_count"),
    #CustomCspTransformer2(mode="low_electrodes_count"),
    TimeVaeTransformer(encoder_output = 3, 
                       vae_iterations = 5, 
                       use_augmented_data = False, 
                       use_pretrained_scaler = True,
                       vae_latent_dim = 20,
                       ),
    LDA()
)

# pipelines["TimeVAE+LDA_35T"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, vae_iterations = 5, use_augmented_data = True),
#     LDA()
# )

#good also
# pipelines["TimeVAE+LDA_310F"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, vae_iterations = 10, use_augmented_data = False),
#     LDA()
# )

# pipelines["TimeVAE+LDA_310T"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=3, vae_iterations = 10, use_augmented_data = True),
#     LDA()
# )
#=========================================================================

#bad
# pipelines["TimeVAE+LDA_210F"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=2, vae_iterations = 10, use_augmented_data = False),
#     LDA()
# )

# pipelines["TimeVAE+LDA_210T"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=2, vae_iterations = 10, use_augmented_data = True),
#     LDA()
# )
#=================================================================
# pipelines["TimeVAE+LDA_110F"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=1, vae_iterations = 10, use_augmented_data = False),
#     LDA()
# )

#best so far
# pipelines["TimeVAE+LDA_010F"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=0, vae_iterations = 10, use_augmented_data = False),
#     LDA()
# )

#good in some respects
# pipelines["TimeVAE+LDA_010T"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=0, vae_iterations = 10, use_augmented_data = True),
#     LDA()
# )

# pipelines["TimeVAE+LDA_020T"] = make_pipeline(
#     Covariances("oas"),
#     #CustomCspTransformer2(mode="high_electrodes_count"),
#     #CustomCspTransformer2(mode="low_electrodes_count"),
#     TimeVaeTransformer(encoder_output=0, vae_iterations = 20, use_augmented_data = True),
#     LDA()
# )

#different classifiers, different encoder_outputs, with/without data augmentation

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

t1_start = perf_counter() 
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
t1_stop = perf_counter()
print("Elapsed time during the benchmark:", (t1_stop-t1_start) / 60)
print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean()[["score", "time"]])

# save results
save_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "results_dataframe_test.csv"
)
results.to_csv(save_path, index=True)

print("Building statistic plots")
plot_stat(results)