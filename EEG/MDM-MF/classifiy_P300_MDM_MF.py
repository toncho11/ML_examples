"""
====================================================================
Classification of P300 datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

"""
# Author: Anton Andreev
# Modified from plot_classify_EEG_tangentspace.py of pyRiemann
# License: BSD (3-clause)

from pyriemann.estimation import XdawnCovariances, ERPCovariances, Covariances
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
from moabb import set_log_level
from moabb.datasets import (
    bi2013a,
    BNCI2014008,
    BNCI2014009,
    BNCI2015003,
    EPFLP300,
    Lee2019_ERP,
    bi2014a,
    bi2015a
)
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import P300
from pyriemann.classification import MDM
from enchanced_mdm_mf import MeanField
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm

print(__doc__)

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore")

set_log_level("info")

##############################################################################
# Create Pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.

##############################################################################
# We have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {"Target": 1, "NonTarget": 0}

paradigm = P300(resample=128,fmin=1, fmax=24)

#name,    electrodes,   subjects
#bi2013a	      16	24 (normal)                    
#bi2014a    	  16	64 (usually low performance)
#BNCI2014009	  16	10 (usually high performance)
#BNCI2014008	   8	 8
#BNCI2015003	   8	10
#bi2015a          32    43
#bi2015b          32    44
   
datasets = [bi2015a()] #bi2014a(),
#datasets = [bi2013a(), BNCI2014008(), BNCI2014009(),BNCI2015003()] #bi2014a(),

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
# n_subjects = 3
# for dataset in datasets:
#     dataset.subject_list = dataset.subject_list[0:n_subjects]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

power_means = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
#power_means = [-1, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1]

# pipelines["XD+MDM_MF_3PM"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     MeanField(power_list=[-1, 0, 1],
#               method_label="inf_means"),
# )

pipelines["XD+MDM_MF_10PM"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    #sum_means does not make a difference with 10 power means comapred to 3
    MeanField(power_list=power_means,
              method_label="inf_means",
              n_jobs=12,
              ),
)

pipelines["XD+MDM_MF_10PM_LDA"] = make_pipeline(
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

# pipelines["XD+MDM_MF_10PM_GPR"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=12,
#              ),
#     GaussianProcessRegressor(alpha = 0.1, kernel = RBF(length_scale_bounds = (0.1, 1.0)))
# )

# pipelines["XD+MDM_MF_10PM_SVM"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=12,
#              ),
#     svm.SVC(kernel="rbf")
# )

# this is a non quantum pipeline
pipelines["XD+MDM"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    MDM(),
)

print("Total pipelines to evaluate: ", len(pipelines))

evaluation = CrossSubjectEvaluation(
    paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
)

results = evaluation.process(pipelines)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

##############################################################################
# Plot Results
# ----------------
#
# Here we plot the results to compare the two pipelines

fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

sns.stripplot(
    data=results,
    y="score",
    x="pipeline",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)
sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1")

ax.set_ylabel("ROC AUC")
ax.set_ylim(0.3, 1)

plt.show()
