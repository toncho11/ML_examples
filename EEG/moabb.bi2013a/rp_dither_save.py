from pyriemann.estimation import Covariances, ERPCovariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from moabb.datasets import bi2013a, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300

import numpy as np

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mne import set_log_level
set_log_level("CRITICAL")

datasets = [bi2013a()] # , EPFLP300(), BNCI2015003(), BNCI2014008(), BNCI2014009()]
paradigm = P300()

le = LabelEncoder()

cov_estimator = XdawnCovariances(classes=[1], estimator='oas', xdawn_estimator='oas')
ts = TangentSpace()
clf = LogisticRegression(solver="liblinear")
pipeline = make_pipeline(cov_estimator, ts, clf)

scorer = make_scorer(balanced_accuracy_score)
for dataset in datasets:
    for source_i, source in enumerate(dataset.subject_list):
        X, y, _ = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:2]) #dataset.subject_list[:10])#
        y = le.fit_transform(y)
        scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer, n_jobs=-1)
        print(scores)
        print(np.mean(scores))
