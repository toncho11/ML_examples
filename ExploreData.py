# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:02:48 2022

@author: antona

This script loads previously saved data in the format (samples, features).
It plots the data, performs clustering and classification. 
It gives a basic idea if classes are separable.
It is designed for two classes with labels 0 and 1

"""

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import xgboost as xgb

#import Dither #pip install PyDither
import os
import glob
from time import time
import sys
import gc

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def LoadTrainTest():
    filename = 'C:\\Work\\PythonCode\\ML_examples\\EEG\\DataAugmentation\\UsingTimeVAE\\TrainTest.npz'
    print("Loading data from: ", filename)
    data = np.load(filename)
    
    return data['X_train'] , data['X_test'], data['y_train'], data['y_test']
    
#Plot mean for each class
def PlotAverage(X_train, X_test, y_train, y_test):
    
    from matplotlib import pyplot as plt 
    
    plt.subplot(211)
   
    #plot class 0 Train
    indices = y_train[y_train == 0]
    class0 = X_train[indices]
    average = np.average(class0, axis=0)
    plt.plot(average)
    
    #plot class 1 Train
    indices = y_train[y_train == 1]
    class1 = X_train[indices]
    average = np.average(class1, axis=0)
    plt.plot(average)
    
    #plt.set_title("Axis 1 title")
    #plt.set_xlabel("X-label for axis 1")
    
    plt.subplot(212) # two axes on figure
    
    #plot class 0 Test
    indices = y_test[y_test == 0]
    class0 = X_test[indices]
    average = np.average(class0, axis=0)
    plt.plot(average)
    
    #plot class 1 Test
    indices = y_test[y_test == 1]
    class1 = X_test[indices]
    average = np.average(class1, axis=0)
    plt.plot(average)

def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    
#Perform KMeans clustering
def KMeansClustering(X_train, X_test, y_train, y_test):
    
    print(82 * "_")
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    
    srategy = "k-means++"
    kmeans = KMeans(init=srategy, n_clusters=X_train.shape[1], n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name=srategy, data=X_train, labels=y_train)
    
    pred = kmeans.fit_predict(X_train)
    PlotCluster(X_train, pred)
    
    srategy = "random"
    kmeans = KMeans(init=srategy, n_clusters=X_train.shape[1], n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name=srategy, data=X_train, labels=y_train)
    
    pred = kmeans.fit_predict(X_train)
    PlotCluster(X_train, pred)
    
    # using initialization strategy where the centers are provided by another algorithm such as PCA
    pca = PCA(n_components=X_train.shape[1]).fit(X_train)
    #pca.components_ is the set of all eigenvectors (aka loadings) for the projection space (one eigenvector for each principal component).
    #Kmeans init requires (n_clusters, n_features) type of input n_clusters = n_components in PCA
    #and n_features = the size of the eigen vector in PCA to be used as a feature vector.
    indices = [4, 7]
    kmeans = KMeans(init=pca.components_[indices,:], n_clusters=2, n_init=1)
    bench_k_means(kmeans=kmeans, name="PCA-based", data=X_train, labels=y_train)
    
    pred = kmeans.fit_predict(X_train)
    PlotCluster(X_train, pred)
    
def PlotCluster(df, label):
    
    #filter rows of original data
    filtered_label2 = df[label == 0]
     
    filtered_label8 = df[label == 1]
     
    #Plotting the results
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
    plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
    plt.show()

    
#Perform DBSCAN clustering
def DbscanClustering(X_train, X_test, y_train, y_test):
    
    X = X_train
    labels_true = y_train 
    
    db = DBSCAN(eps=6, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % adjusted_rand_score(labels_true, labels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % adjusted_mutual_info_score(labels_true, labels)
    )
    #print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
    
    pred = db.fit_predict(X_train)
    PlotCluster(X_train, pred)

#Classify with SVM
def EvaluateSVM(X_train, X_test, y_train, y_test):
    
    from sklearn.svm import LinearSVC, SVC

    clf = SVC(C=1.0, random_state=1, kernel='rbf', verbose=False)
    #clf = LinearSVC(C=1.0, random_state=1, dual=False, verbose=False)
 
    # Fit the model
    print("Training standard classifier ...")
    clf.fit(X_train, y_train)

    print("Predicting standard classifier ...")
    y_pred = clf.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy SVM #####: ", ba)
    print("Accuracy score    SVM #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     SVM #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s     P300: ", sum(y_pred), "/", sum(y_test))
    print("0s Non P300: ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return cr, ba, clf

#Classify with Neural Network
def EvalauteNN(X_train, X_test, y_train, y_test, epochs):
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential([
      # Dense(24, activation=tf.nn.relu,input_shape=(X_train.shape[1],)),
      # Dense(12, activation=tf.nn.relu),
      # Dense(1,  activation=tf.nn.sigmoid)
      Dense(8, activation=tf.nn.relu,input_shape=(X_train.shape[1],)),
      Dense(4, activation=tf.nn.relu),
      Dense(1,  activation=tf.nn.sigmoid)
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
        verbose=False,
        validation_data=(X_test, y_test), #not good because you have a glimpse on the final test dataset
        )
    
    y_pred = model.predict(X_test)
    
    y_pred = y_pred.round()
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy NN #####: ", ba)
    print("Accuracy score    NN #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     NN #####: ", roc_auc_score(y_test, y_pred))
    
    return ba

#Classify with Boosting using XGBoost
def EvalauteXGBoost(X_train, X_test, y_train, y_test):
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    
    xg_reg.fit(X_train,y_train)
    
    y_pred = xg_reg.predict(X_test)
    
    y_pred = y_pred.round()
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy XGBoost #####: ", ba)
    print("Accuracy score    XGBoost #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     XGBoost #####: ", roc_auc_score(y_test, y_pred))
    
    return ba

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = LoadTrainTest()
    print(X_train.shape)
    
    #PlotAverage(X_train, X_test, y_train, y_test)
    
    #DbscanClustering(X_train, X_test, y_train, y_test) #it needs setting two parameters manually
    
    KMeansClustering(X_train, X_test, y_train, y_test)
    
    #EvaluateSVM(X_train, X_test, y_train, y_test)
    
    #EvalauteNN(X_train, X_test, y_train, y_test, 30)
    
    #EvalauteXGBoost(X_train, X_test, y_train, y_test)