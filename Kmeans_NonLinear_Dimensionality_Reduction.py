# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:28:48 2022

source: https://medium.com/analytics-vidhya/less-known-applications-of-k-means-clustering-dimensionality-reduction-anomaly-detection-and-908f4bee155f#:~:text=Non%20Linear%20Dimensionality%20Reduction%20using,number%20of%20clusters%20to%202.

Using KMeans for dimensionality reduction and then for classification.
New features are created based on the distance between each point and the centroids calculated by Kmeans.
This improves classification 2%-3%.

@author: antona
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model, preprocessing, model_selection, pipeline, ensemble, tree, datasets, cluster
from sklearn.cluster import KMeans

sns.set(style = 'white', font_scale = 1.4)
### Load the Data
data  =  pd.DataFrame(datasets.load_boston().data, columns = datasets.load_boston().feature_names)
y = datasets.load_boston().target

features = ['CRIM', 'LSTAT', 'RM', 'AGE', 'INDUS', 'NOX', 'DIS']
data = data[features] #use only selected features

# Scale and Fit KMeans to data as part of a pipeline
# transform() calculates the distance of each data point from each cluster center.
def ScatterPlotCentroidDistance(data):
    kmeans = pipeline.make_pipeline(preprocessing.StandardScaler(), cluster.KMeans(n_clusters = 2)).fit(X_train)
    lower_dim = pd.DataFrame(kmeans.transform(data), columns = ['Comp 1', 'Comp 2']) #we have two clusters and thus two columns
    lower_dim.plot.scatter( x='Comp 1',y= 'Comp 2', grid = True, figsize = (10, 7))

### Train Test Split
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y, test_size = .2, random_state = 10)
# note that X_test and y_test are not used in the pipeline bleow

ScatterPlotCentroidDistance(X_train)

### Fit a Linear Model
model = linear_model.LinearRegression()
score = model_selection.cross_val_score(model, X_train, y_train, cv = 10, scoring = 'r2')
print(f'Average LR r2: {np.mean(score)}')

### Perform the transformation using K-Means and train a linear model on the transformed features
km = pipeline.make_pipeline(preprocessing.StandardScaler(), cluster.KMeans(n_clusters = 5),
                           linear_model.LinearRegression())

score = model_selection.cross_val_score(km, X_train, y_train, cv = 10, scoring = 'r2')
print(f'Average KMeans + LR r2: {np.mean(score)}')