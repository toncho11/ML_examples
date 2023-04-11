# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:27:57 2023

@author: antona

Usually the amount of P300 samples is much lower than the Non-P300 samples which
creates a highly imbalanced classes.
Here the BalancedBaggingClassifier is used. It is a Bagging classifier with additional balancing.

It can be used in two modes (see below):
    - CrossSubjectEvaluation
    - WithinSessionEvaluation
"""

import numpy as np
import sys
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.pipeline import make_pipeline
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import bi2013a #, bi2014a, bi2014b, bi2015a, bi2015b, BNCI2014008, BNCI2014009, BNCI2015003, EPFLP300, Lee2019_ERP
from moabb.paradigms import P300
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Rescaling
from tensorflow import map_fn
import tensorflow as tf
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
)

paradigm = P300()
    
if __name__ == "__main__":
    
    #warning when usiung multiple datasets they must have the same number of electrodes 
    
    # CONFIGURATION
    #https://github.com/toncho11/ML_examples/wiki/EEG-datasets
    #name, electrodes, subjects
    #bi2013a	    16	24 (normal)
    #bi2014a    	16	64 (usually low performance)
    #BNCI2014009	16	10 (usually high performance)
    #BNCI2014008	 8	 8
    #BNCI2015003	 8	10
    #bi2015a        32  43
    #bi2015b        32  44
    #ds = [bi2014a(), bi2013a()] #both 16ch, 512 freq
    #ds = [bi2015a(), bi2015b()] #both 32ch, 512 freq
    n = 20
    ds = [bi2013a()] #Warning all datasets different from BNCI2014009 have too big epochs to be fit in the video memory
    #epochs = 45 #default 60
    xdawn_filters_all = 4 #default 4
    
    # init
    pure_mdm_scores = []
    tf_scores = []
    
    #create pipelines
    pipelines = {}
    pipelines["MDM"] = make_pipeline(XdawnCovariances(xdawn_filters_all), MDM())
    pipelines["TS+BBC"] =  make_pipeline(
        XdawnCovariances(xdawn_filters_all),
        TangentSpace(),
        BalancedBaggingClassifier(
          estimator=HistGradientBoostingClassifier(random_state=42),
          n_estimators=10,
          random_state=42,
          n_jobs=2,
      ))
    
    # from sklearn.svm import SVC
    # pipelines["TS+SVC"] =  make_pipeline(
    #     XdawnCovariances(xdawn_filters_all),
    #     TangentSpace(),
    #     SVC()
    #   )
    
    # from lightgbm import LGBMClassifier
    # pipelines["TS+LGBM"] =  make_pipeline(
    #     XdawnCovariances(xdawn_filters_all),
    #     TangentSpace(),
    #     LGBMClassifier()
    #  )
    
    print("Processing ... ")
    #CrossSubjectEvaluation
    #WithinSessionEvaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=ds,
        overwrite=True
    )

    results = evaluation.process(pipelines)
    
    print("Averaging the session performance:")
    print("Type of metric:", paradigm.scoring)
    print(results.groupby('pipeline').mean('score')[['score', 'time']])
   
