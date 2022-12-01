# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 09:56:05 2022

source: https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn

The aim of this guide is to build a classification model to detect diabetes.
This is a modified example from the above link that uses TensorFlow Keras instead of 
scikit learn for the implementation of the neural network.
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
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from sklearn.metrics import balanced_accuracy_score, make_scorer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def EvalauteNN(X_train, X_test, y_train, y_test, epochs):
    model = Sequential([
      Dense(8, activation=tf.nn.relu,input_shape=(X_train.shape[1],)),
      Dense(8, activation=tf.nn.relu),
      Dense(8, activation=tf.nn.relu),
      Dense(1, activation=tf.nn.sigmoid)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        )
    
    print("Training NN ...")
    model.fit(
        X_train, # training data
        y_train, # training targets
        epochs=epochs, #how long to train
        batch_size=32,
        verbose=True,
        validation_data=(X_test, y_test),
        )
    
    y_pred_test = model.predict(X_test)
    
    y_pred_test = y_pred_test.round()
    
    predict_train = model.predict(X_train)
    
    predict_train = predict_train.round()
    
    ba = balanced_accuracy_score(y_test, y_pred_test)
    print("Balanced Accuracy #####: ", ba)
    print("Accuracy score    #####: ", sklearn.metrics.accuracy_score(y_test, y_pred_test))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     #####: ", roc_auc_score(y_test, y_pred_test))
    
    print(confusion_matrix     (y_train, predict_train))
    print(classification_report(y_train, predict_train))
    
    return ba

df = pd.read_csv('../diabetes.csv')  #kaggle ('out' = 'diabetes')
print(df.shape)
df.describe().transpose()

target_column = ['diabetes'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

EvalauteNN(X_train, X_test, y_train, y_test, 100)