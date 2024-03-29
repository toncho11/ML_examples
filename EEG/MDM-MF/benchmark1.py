"""
====================================================================
Classification of P300 datasets from MOABB using MDM-MF
====================================================================

MDM-MF is the Riemammian Mimimum Distance to Means Field Classifier
Paper: https://hal.science/hal-02315131

Warning:
Use MOABB 1.0 with https://github.com/NeuroTechX/moabb/issues/514 fixed.
Otherwise evaluation silently fails on multiple datasets.

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
from moabb.paradigms import P300, MotorImagery, LeftRightImagery
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
labels_dict = {"Target": 1, "NonTarget": 0}

overwrite = False  # set to True if we want to overwrite cached results

cache_config = dict(
    use=True,
    save_raw=False,
    save_epochs=True,
    save_array=False,
    overwrite_raw=False,
    overwrite_epochs=False,
    overwrite_array=False,
)

paradigm_P300 = P300()#(resample=128,fmin=1, fmax=24)
paradigm_MI   = MotorImagery()#(fmin=8,fmax=32)
paradigm_LR   = LeftRightImagery()#(fmin=8,fmax=32)

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
datasets_P300 = [BI2013a(), BNCI2014_008(), BNCI2014_009(), BNCI2015_003()] #, EPFLP300()
#original 12 ds for MI/LR 
datasets_MI = [ #BNCI2015_004(), #5 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
                BNCI2015_001(),  #2 classes
                BNCI2014_002(),  #2 classes
                #AlexMI(),       #3 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
              ]

datasets_LR = [ BNCI2014_001(),
                BNCI2014_004(),
                Cho2017(),      #49 subjects
                GrosseWentrup2009(),
                PhysionetMI(),  #109 subjects
                Shin2017A(accept=True), 
                Weibo2014(), 
                Zhou2016(),
              ]

#each MI dataset can have different classes and events and this requires a different MI paradigm
paradigms_MI = []
for dataset in datasets_MI:
    events = list(dataset.event_id)
    paradigm = MotorImagery(events=events, n_classes=len(events))
    paradigms_MI.append(paradigm)

#checks if correct paradigm is used
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
        
for d in datasets_LR:
    name = type(d).__name__
    print(name)
    if name not in [(lambda x: type(x).__name__)(x) for x in paradigm_LR.datasets]:
        print("Error: dataset not compatible with selected paradigm", name)
        import sys
        sys.exit(1)
        

# adjust the number of subjects, the Quantum pipeline takes a lot of time
# if executed on the entire dataset
max_n_subjects = 30
for dataset in datasets_P300:
    n_subjects_ds = min(max_n_subjects,len(dataset.subject_list))
    dataset.subject_list = dataset.subject_list[0:n_subjects_ds]
    
for dataset in datasets_MI:
    n_subjects_ds = min(max_n_subjects,len(dataset.subject_list))
    # name = type(dataset).__name__
    # if (name == "BNCI2015_004"): #remove first subject
    #     dataset.subject_list = dataset.subject_list[1:n_subjects_ds]
    # else:
    dataset.subject_list = dataset.subject_list[0:n_subjects_ds]
    
for dataset in datasets_LR:
    n_subjects_ds = min(max_n_subjects,len(dataset.subject_list))
    dataset.subject_list = dataset.subject_list[0:n_subjects_ds]

pipelines = {}

#power_means = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
#power_means = [-1, -0.8, -0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
#original p
power_means = [-1, -0.75, -0.5, -0.25, -0.1, -0.01, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

pipelines["MDM_MF"] = make_pipeline(
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

pipelines["MDM_MF_LDA"] = make_pipeline(
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

pipelines["MDM_MF_LR_l1"] = make_pipeline(
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
    LogisticRegression(penalty="l1", solver="liblinear")
)

pipelines["MDM_MF_LR_l2"] = make_pipeline(
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
    LogisticRegression(penalty="l2", solver="lbfgs")
)

pipelines["XD+MDM_MF_GPR"] = make_pipeline(
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
    GaussianProcessRegressor(alpha = 0.1, kernel = RBF(length_scale_bounds = (0.1, 1.0)))
)

pipelines["MDM_MF_SVM"] = make_pipeline(
    # applies XDawn and calculates the covariance matrix, output it matrices
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    MeanField(power_list=power_means,
              method_label="sum_means", #not used if used as transformer
              n_jobs=12,
              ),
    svm.SVC(kernel="rbf")
)

# this is a non quantum pipeline
pipelines["MDM"] = make_pipeline(
    XdawnCovariances(
        nfilter=3,
        classes=[labels_dict["Target"]],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    MDM(),
)

print("Total pipelines to evaluate: ", len(pipelines))

evaluation_P300 = WithinSessionEvaluation(
    paradigm=paradigm_P300, datasets=datasets_P300, suffix="examples", overwrite=overwrite,
    n_jobs=12,
    n_jobs_evaluation=12,
    cache_config=cache_config
)
evaluation_LR = WithinSessionEvaluation(
    paradigm=paradigm_LR, datasets=datasets_LR, suffix="examples", overwrite=overwrite,
    n_jobs=12,
    n_jobs_evaluation=12,
    cache_config=cache_config
)

results_P300 = evaluation_P300.process(pipelines)

#replace XDawnCovariances with Covariances when using MI or LeftRightMI
for pipe_name in pipelines:
    pipeline = pipelines[pipe_name]
    pipeline.steps.pop(0)
    pipeline.steps.insert(0,['covariances',Covariances('oas')])

results_LR = evaluation_LR.process(pipelines)

results = pd.concat([results_P300, results_LR],ignore_index=True)

#each MI dataset uses its own configured MI paradigm
for paradigm_MI, dataset_MI in zip(paradigms_MI, datasets_MI):
    evaluation_MI = WithinSessionEvaluation(
        paradigm=paradigm_MI,
        datasets=[dataset_MI],
        overwrite=overwrite,
        n_jobs=12,
        n_jobs_evaluation=12,
        cache_config=cache_config
    )
    results_per_MI_pardigm = evaluation_MI.process(pipelines)
    results = pd.concat([results, results_per_MI_pardigm],ignore_index=True)

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