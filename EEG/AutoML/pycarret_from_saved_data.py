# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:10:24 2022

This example shows how to load previously processed data stored as npz and use the AutoML lib PyCarret on it.

pip install pycaret

@author: antona
"""

from pycaret.classification import *
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import *

def LoadTrainTest():
    filename = 'C:\\Work\\PythonCode\\ML_examples\\EEG\\DataAugmentation\\UsingTimeVAE\\TrainTest.npz'
    print("Loading data from: ", filename)
    data = np.load(filename)
    
    return data['X_train'] , data['X_test'], data['y_train'], data['y_test']

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = LoadTrainTest()
    
    print(X_train.shape)
    
    X_train_y_df = pd.DataFrame(X_train)
    X_train_y_df["y"] = y_train.tolist()
    
    s = setup(X_train_y_df, target = "y", verbose = True, silent=True)
    
    best_model = compare_models()
    print("Best model: ", best_model)
    
    y_pred = best_model.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy Test data #####: ", ba)
    print("Accuracy score    Test data #####: ", accuracy_score(y_test, y_pred))
    print("ROC AUC score     Test data #####: ", roc_auc_score(y_test, y_pred))
   
    print("1s : ", sum(y_pred), "/", sum(y_test))
    print("0s : ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))

    