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
from sklearn import decomposition

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
from keras.utils import to_categorical

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
    
    PlotCluster(X_train, y_train, "real")
    
    srategy = "k-means++"
    kmeans = KMeans(init=srategy, n_clusters=X_train.shape[1], n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name=srategy, data=X_train, labels=y_train)
    
    pred = kmeans.fit_predict(X_train)
    PlotCluster(X_train, pred, "k-means++")
    
    # pred = kmeans.fit_predict(X_test)
    # PlotClusterReal(X_test, pred, "k-means++ pred", y_test)
    
    srategy = "random"
    kmeans = KMeans(init=srategy, n_clusters=X_train.shape[1], n_init=4, random_state=0)
    bench_k_means(kmeans=kmeans, name=srategy, data=X_train, labels=y_train)
    
    pred = kmeans.fit_predict(X_train)
    PlotCluster(X_train, pred, "random")
    
    # using initialization strategy where the centers are provided by another algorithm such as PCA
    pca = PCA(n_components=X_train.shape[1]).fit(X_train)
    #pca.components_ is the set of all eigenvectors (aka loadings) for the projection space (one eigenvector for each principal component).
    #Kmeans init requires (n_clusters, n_features) type of input n_clusters = n_components in PCA
    #and n_features = the size of the eigen vector in PCA to be used as a feature vector.
    
    # for i in range(0,X_train.shape[1]-2):
    #     indices = [i, i+2]
    #     kmeans = KMeans(init=pca.components_[indices,:], n_clusters=2, n_init=1)
    #     bench_k_means(kmeans=kmeans, name="PCA-based", data=X_train, labels=y_train)
    
    #     pred = kmeans.fit_predict(X_test)
    #     PlotCluster(X_test, pred, "PCA-based")
        
    # for i in range(0,X_train.shape[1]-5):
    #     indices = [i, i+5]
    #     kmeans = KMeans(init=pca.components_[indices,:], n_clusters=2, n_init=1)
    #     bench_k_means(kmeans=kmeans, name="PCA-based", data=X_train, labels=y_train)
    
    #     pred = kmeans.fit_predict(X_test)
    #     PlotCluster(X_test, pred, "PCA-based")
    
def PlotCluster(X, label, title):
    
    #filter rows of original data
    filtered_label2 = X[label == 0]
     
    filtered_label8 = X[label == 1]
     
    #Plotting the results
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
    plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'blue')
    plt.title(title)
    plt.show()

    
def PlotClusterReal(X, y_pred, title, y_real):
    
    #filter rows of original data
    filtered_label2 = X[(y_pred == 0) & (y_real==0)]
     
    filtered_label8 = X[(y_pred == 1) & (y_real==1)]
    
    rest_indices = (((y_pred == 0) & (y_real==0)) | ((y_pred == 1) & (y_real==1)))
    rest = X[~rest_indices]
    
    #Plotting the results
    plt.scatter(rest[:,0] , rest[:,1] , color = 'black') # not correctly 
    
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
    plt.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'blue')
    
    
    plt.title(title)
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
    print("Training SVM ...")
    clf.fit(X_train, y_train)

    print("Predicting SVM ...")
    y_pred = clf.predict(X_test)
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy SVM #####: ", ba)
    print("Accuracy score    SVM #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     SVM #####: ", roc_auc_score(y_test, y_pred))
    
    print("1s : ", sum(y_pred), "/", sum(y_test))
    print("0s : ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
    from sklearn.metrics import classification_report
    cr = classification_report(y_test, y_pred, target_names=['Non P300', 'P300'])
    #print(cr)
    return cr, ba, clf

#Classify with Neural Network
def EvalauteNN(X_train, X_test, y_train, y_test, epochs):
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    layer_1N = X_train.shape[1]
    layer_2N = round(X_train.shape[1] / 2)
    print("Layers:", layer_1N, layer_2N)
    
    model = Sequential([
      # Dense(24, activation=tf.nn.relu,input_shape=(X_train.shape[1],)),
      # Dense(12, activation=tf.nn.relu),
      # Dense(1,  activation=tf.nn.sigmoid)
       Dense(layer_1N, activation=tf.nn.relu,input_shape=(X_train.shape[1],)),
       Dense(layer_2N, activation=tf.nn.relu),
       Dense(1,  activation=tf.nn.sigmoid)
      # Dense(2, activation=tf.nn.relu, input_shape=(1,)),
      # Dense(1, activation=tf.nn.sigmoid)
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
        validation_data=(X_test, y_test), #not good because you have a glimpse on the final test dataset
        )
    
    y_pred = model.predict(X_test)
    
    y_pred = y_pred.round()
    
    ba = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Accuracy NN #####: ", ba)
    print("Accuracy score    NN #####: ", accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_auc_score
    print("ROC AUC score     NN #####: ", roc_auc_score(y_test, y_pred))
    print("1s : ", sum(y_pred), "/", sum(y_test))
    print("0s : ", len(y_pred) - sum(y_pred) , "/", len(y_test) - sum(y_test))
    
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

# First we use KMeans in unsupervised manner to generate clusters.
# Then these clusters are used as feature vectors for the NN 
def EvalauteKMeansNN(X_train, X_test, y_train, y_test):
    
    srategy = "k-means++"
    
    ba_accuracy = []
    
    for n in range(2,20):
        
        print("n =",n)
        
        clfKM = KMeans(init=srategy, n_clusters=n, n_init=4)
        
        clfKM.fit(X_train)
        
        y_pred_km_train = clfKM.predict(X_train)#.reshape(X_train.shape[0],1)
        
        y_pred_km_train = to_categorical(y_pred_km_train, dtype ="uint8")
        
        y_pred_km_test  = clfKM.predict(X_test )#.reshape(X_test.shape[0] ,1)
        
        y_pred_km_test  = to_categorical(y_pred_km_test, dtype ="uint8")
        
        ba = EvalauteNN(y_pred_km_train, y_pred_km_test, y_train, y_test, 50) 
        
        ba_accuracy.append(ba)
    
    print(ba_accuracy)
    
if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = LoadTrainTest()
    print(X_train.shape)
    
    # Apply PCA
    # pca = decomposition.PCA(n_components=0.95)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # print(X_train.shape[1])
    # X_test  = pca.transform(X_test)
    
    # Feature selection
    # from sklearn.feature_selection import VarianceThreshold
    # sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    # X_train = sel.fit_transform(X_train)
    # X_test  = sel.fit_transform(X_test)
    # print(X_train.shape)
    
    #PlotAverage(X_train, X_test, y_train, y_test)
    
    #DbscanClustering(X_train, X_test, y_train, y_test) #it needs setting two parameters manually
    
    KMeansClustering(X_train, X_test, y_train, y_test)
    
    #EvaluateSVM(X_train, X_test, y_train, y_test)
    
    #EvalauteNN(X_train, X_test, y_train, y_test, 100)
    
    #the Kmeans must detect the classes well otherwise it won't work
    #EvalauteKMeansNN(X_train, X_test, y_train, y_test)
    
    #EvalauteXGBoost(X_train, X_test, y_train, y_test)