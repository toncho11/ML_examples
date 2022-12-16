# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:54:44 2022

@author: antona
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'], random_state=42)
            

def LoadTrainTest():
    filename = filename = 'C:\\Work\\PythonCode\\ML_examples\\EEG\\DataAugmentation\\UsingTimeVAE\\data\\TrainTest_augmentation_MDMD_77.npz'
    print("Loading data from: ", filename)
    data = np.load(filename)
    
    return data['X_train'] , data['X_test'], data['y_train'], data['y_test']

if __name__ == "__main__":

    X_train, X_test, y_train, y_test =  LoadTrainTest()

    #START PASTE HERE
    #Average CV score on the training set was: 0.8030163599182003
    exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.6000000000000001, min_samples_leaf=3, min_samples_split=16, n_estimators=100)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 42)
    #END PASTE HERE
    
    exported_pipeline.fit(X_train, y_train)
    
    y_pred = exported_pipeline.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy Test data #####: ", ba)
    print("Accuracy score    Test data #####: ", accuracy_score(y_test, y_pred))
    print("ROC AUC score     Test data #####: ", roc_auc_score(y_test, y_pred))
   
    print("1s : ", sum(y_pred), "/", sum(y_test))
    print("0s : ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))