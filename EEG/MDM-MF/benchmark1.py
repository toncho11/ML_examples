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
import pandas as pd
from moabb import set_log_level
#P300
from moabb.datasets import (
    BI2013a,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_003,
    EPFLP300,
    Lee2019_ERP,
    bi2014a,
    bi2015a,
    bi2015b,
    EPFLP300
)
#Motor imagery
from moabb.datasets import (
    BNCI2014_001, 
    Zhou2016, 
    BNCI2015_001, 
    BNCI2014_002, 
    BNCI2014_004, 
    BNCI2015_004, 
    AlexMI, 
    Weibo2014, 
    Cho2017, 
    GrosseWentrup2009, 
    PhysionetMI, 
    Shin2017A
)
from moabb.evaluations import WithinSessionEvaluation, CrossSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import P300, MotorImagery
from pyriemann.classification import MDM
from enchanced_mdm_mf import MeanField
from pyriemann.classification import MeanField as MeanField_orig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import svm

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
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
#labels_dict = {"Target": 1, "NonTarget": 0}

paradigm_P300 = P300(resample=128,fmin=1, fmax=24)
paradigm_MI   = MotorImagery(fmin=8,fmax=32)

#name,    electrodes,   subjects
#bi2013a	      16	24 (normal)                    
#bi2014a    	  16	64 (usually low performance)
#BNCI2014009	  16	10 (usually high performance)
#BNCI2014008	   8	 8
#BNCI2015003	   8	10
#bi2015a          32    43
#bi2015b          32    44
   
#datasets = [bi2013a(),BNCI2014008()] #bi2014a(),
#datasets = [bi2013a(), BNCI2014008(), BNCI2014009(),BNCI2015003(), bi2014a()]
#datasets = [bi2013a(), BNCI2014008(), BNCI2014009(),BNCI2015003(), bi2014a(), bi2015b()]

#original 5 ds for P300
datasets_P300 = [BI2013a()]#, BNCI2014_008(), BNCI2014_009(), BNCI2015_003(), EPFLP300()]
#original 12 ds for MI 
#BNCI2015_001(), #not working "did not have enough events in None to run analysis"
#BNCI2014_002(), #not working "did not have enough events in None to run analysis"
#BNCI2014_004(), #not working "did not have enough events in None to run analysis"
datasets_MI = [ BNCI2015_004(),
                #BNCI2015_001(),
                #Zhou2016(),
                #BNCI2014_001(),
                #BNCI2014_002(),    
                #BNCI2014_004()
                #AlexMI(), 
                Weibo2014(), 
                #Cho2017(), 
                PhysionetMI(), 
                #Shin2017A(), 
                #GrosseWentrup2009()
                ]

#checks
for d in datasets_P300:
    name = type(d).__name__
    print(name)
    if name not in [(lambda x: type(x).__name__)(x) for x in paradigm_P300.datasets]:
        print("Error: dataset not compatible with selected paradigm", name)
        import sys
        sys.exit(1)
        
for d in datasets_MI:
    name = type(d).__name__
    print(name)
    if name not in [(lambda x: type(x).__name__)(x) for x in paradigm_MI.datasets]:
        print("Error: dataset not compatible with selected paradigm", name)
        import sys
        sys.exit(1)
        

# reduce the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
n_subjects = 1
for dataset in datasets_P300:
    dataset.subject_list = dataset.subject_list[0:n_subjects]
for dataset in datasets_MI:
    dataset.subject_list = dataset.subject_list[0:n_subjects]

overwrite = True  # set to True if we want to overwrite cached results

pipelines = {}

#power_means = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
#power_means = [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
#original p
power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

# pipelines["MDM_MF"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     #sum_means does not make a difference with 10 power means comapred to 3
#     MeanField_orig(power_list=power_means,
#               method_label="inf_means",
#               n_jobs=1,
#               ),
# )

# pipelines["MDM_MF_LDA"] = make_pipeline(
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
#               ),
#     LDA()
# )

# pipelines["MDM_MF_LR"] = make_pipeline(
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
#               ),
#     LogisticRegression(penalty="l1", solver="liblinear")
# )

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

# this is a non quantum pipeline
pipelines["MDM"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        #classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    MDM(),
)

# pipelines["MDM_MF_SVM"] = make_pipeline(
#     # applies XDawn and calculates the covariance matrix, output it matrices
#     XdawnCovariances(
#         nfilter=3,
#         classes=[labels_dict["Target"]],
#         estimator="lwf",
#         xdawn_estimator="scm",
#     ),
#     MeanField(power_list=power_means,
#               method_label="sum_means", #not used if used as transformer
#               n_jobs=12,
#               ),
#     svm.SVC(kernel="rbf")
# )

print("Total pipelines to evaluate: ", len(pipelines))

evaluation_P300 = WithinSessionEvaluation(
    paradigm=paradigm_P300, datasets=datasets_P300, suffix="examples", overwrite=overwrite
)
evaluation_MI = WithinSessionEvaluation(
    paradigm=paradigm_MI, datasets=datasets_MI, suffix="examples", overwrite=overwrite,
    #return_epochs=True
)

results_P300 = evaluation_P300.process(pipelines)
results_MI   = evaluation_MI.process(pipelines)

results = pd.concat([results_P300, results_MI],ignore_index=True)

print("Results:")
print(results)

print("Averaging the session performance:")
print(results.groupby("pipeline").mean("score")[["score", "time"]])

#save dataframe with results to disk for further analysis

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

#compare two algorithms
#paired_plot, which plots performance in one versus the performance in the other over all chosen datasets
# fig = moabb_plt.paired_plot(results, "MDM_MF_LDA", "MDM")
# plt.show()

#generate statistics for the summary plot
#Compute matrices of p-values and effects for all algorithms over all datasets via combined p-values and
#combined effects methods
stats = compute_dataset_statistics(results)
P, T = find_significant_differences(stats)
#agg = stats.groupby(['dataset']).mean()
#print(agg)
print(stats.to_string()) #not all datasets are in stats

#negative SMD value favors the first algorithm, postive SMD the second
#A meta-analysis style plot that shows the standardized effect with confidence intervals over
#all datasets for two algorithms. Hypothesis is that alg1 is larger than alg2
fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LDA", "MDM")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LDA", "MDM_MF")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM", "MDM_MF")
plt.show()

#summary plot - significance matrix to compare pipelines.
#Visualize significances as a heatmap with green/grey/red for significantly higher/significantly lower.
moabb_plt.summary_plot(P, T)
plt.show()