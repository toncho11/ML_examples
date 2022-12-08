# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:37:43 2022

An example from scikit learn web-site using a dataset of 10 digits.

The number of clusters by default 8 when using scikit learn. It is also a required parameter.

The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, 
minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below).
Description: https://scikit-learn.org/stable/modules/clustering.html#k-means

Kmeans can be intialized by different built-in srategies.
This includes PCA (pca.components_). pca.components_ is not well documented,
but it seems to be K means Init (n_clusters, n_features) = (PCA components, eigen vector)
where you can select which components to use from PCA and what part of the eigen vector to initialize the centroids.
"""

import numpy as np
from sklearn.datasets import load_digits

data, labels = load_digits(return_X_y=True)
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

# using initialization strategy "k-means++"
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

# using initialization strategy "random"
kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

# using initialization strategy where the centers are provided by another algorithm such as PCA
pca = PCA(n_components=n_digits).fit(data)
#pca.components_ is the set of all eigenvectors (aka loadings) for the projection space (one eigenvector for each principal component).
#Kmeans init requires (n_clusters, n_features) type of input n_clusters = n_components in PCA
#and n_features = the size of the eigen vector in PCA to be used as a feature vector.
kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
bench_k_means(kmeans=kmeans, name="PCA-based", data=data, labels=labels)

print(82 * "_")