# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:56:05 2022

source: https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn

The aim of this guide is to build a classification model to detect diabetes.
It uses sklearn MLP Classifier.
"""

# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

def EvaluateNN(X_train, X_test, y_train, y_test):
    
    # Construct NN and train
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train,y_train)

    # Predict ob both Train and Test datasets
    predict_train = mlp.predict(X_train)
    predict_train = predict_train.round()

    predict_test = mlp.predict(X_test)

    y_pred_test = mlp.predict(X_test)
    y_pred_test = y_pred_test.round()

    # Show statistics
    ba = balanced_accuracy_score(y_test, y_pred_test)

    print("Balanced Accuracy #####: ", ba)
    print("Accuracy score    #####: ", sklearn.metrics.accuracy_score(y_test, y_pred_test))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     #####: ", roc_auc_score(y_test, y_pred_test))

    from sklearn.metrics import classification_report,confusion_matrix
    print("Confusion matrix on Train:")
    print(confusion_matrix(y_train,predict_train))
    print(classification_report(y_train,predict_train))
    print("Confusion matrix on Test:")
    print(confusion_matrix(y_test,predict_test))
    print(classification_report(y_test,predict_test))

df = pd.read_csv('diabetes.csv') #kaggle ('out' = 'diabetes')
print(df.shape)
df.describe().transpose()

# convert to X,y
target_column = ['diabetes'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

EvaluateNN(X_train, X_test, y_train, y_test)