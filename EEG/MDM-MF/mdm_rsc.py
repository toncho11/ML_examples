'''

A MDM using random subclassifiewrs

'''

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC as sklearnSVC
from sklearn.utils.extmath import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
import warnings

from pyriemann.utils.kernel import kernel
from pyriemann.utils.mean import mean_covariance, mean_power, mean_logeuclid
from pyriemann.utils.distance import distance
from pyriemann.tangentspace import FGDA, TangentSpace
from pyriemann.utils.distance import distance_euclid
from scipy.stats import zscore
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA,
)

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import copy
from enchanced_mdm_mf_tools import mean_power_custom, distance_custom, power_distance
from time import perf_counter_ns,perf_counter
from pyriemann.clustering import Potato
from pyriemann.classification import MDM
import random

def _check_metric(metric): #in utils in newer versions
    if isinstance(metric, str):
        metric_mean = metric
        metric_dist = metric

    elif isinstance(metric, dict):
        # check keys
        for key in ['mean', 'distance']:
            if key not in metric.keys():
                raise KeyError('metric must contain "mean" and "distance"')

        metric_mean = metric['mean']
        metric_dist = metric['distance']

    else:
        raise TypeError('metric must be dict or str')

    return metric_mean, metric_dist

class MDM_RS(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by Minimum Distance to Mean Field.

    Classification by Minimum Distance to Mean Field [1]_, defining several
    power means for each class.

    Parameters
    ----------
    power_list : list of float, default=[-1,0,+1]
        Exponents of power means.
    method_label : {'sum_means', 'inf_means'}, default='sum_means'
        Method to combine labels:

        * sum_means: it assigns the covariance to the class whom the sum of
          distances to means of the field is the lowest;
        * inf_means: it assigns the covariance to the class of the closest mean
          of the field.
    metric : string, default="riemann"
        Metric used for distance estimation during prediction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : dict of ``n_powers`` lists of ``n_classes`` ndarrays of shape \
            (n_channels, n_channels)
        Centroids for each power and each class.

    See Also
    --------
    MDM

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `The Riemannian Minimum Distance to Means Field Classifier
        <https://hal.archives-ouvertes.fr/hal-02315131>`_
        M Congedo, PLC Rodrigues, C Jutten. BCI 2019 - 8th International
        Brain-Computer Interface Conference, Sep 2019, Graz, Austria.
    """

    def __init__(self, 
                 metric="riemann",
                 n = 5,
                 k = 1
                 ):
        """Init."""
        self.metric = metric 
        self.n = n
        self.k = k
        
        self.mdms = [MDM(metric=self.metric) for i in range(0,self.n)]
      
    def _get_random_data(self,X,y, samples_n, max_samples):
        
        X1 = X.copy()
        y1 = y.copy()
        
        indices = random.sample(range(0, max_samples), samples_n)
        
        X1 = X1[indices,:,:]
        y1 = y1[indices]
        
        return X1,y1
         
    def fit(self, X, y):
        """Fit (estimates) the centroids. Calculates the power means.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray, shape (n_matrices,)
            Labels for each matrix.
        sample_weight : None | ndarray shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : MeanField instance
            The MeanField instance.
        """
        
        #check classes are 0 and 1
        self.classes_ = np.unique(y)
        
        if not (len(self.classes_) == 2 and 0 in self.classes_ and 1 in self.classes_):
            raise Exception("Class labels are not 1 and 0!")
        
        for m in self.mdms:
            Xrs,Yrs = self._get_random_data(X,y, int((X.shape[0] / self.n) * self.k), X.shape[0])
            m.fit(Xrs,Yrs, sample_weight = None)
    
        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest means field.
        """
        results = []
        for x in X:
            
            mdm_result = 0
            
            for m in self.mdms:
                t = np.array(x)
                t = t[np.newaxis,...]
                mdm_result = mdm_result + m.predict(t)[0]
                
            if mdm_result >=3:
                results.append(1)
            else:
                results.append(0)
            
        return results
    
    def predict_proba(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest means field.
        """
        results = []
        for x in X:
            
            mdm_result = 0
            
            for m in self.mdms:
                t = np.array(x)
                t = t[np.newaxis,...]
                mdm_result = mdm_result + m.predict(t)[0]
            
            prob = mdm_result / self.n
            
            #print(prob)
            
            results.append([1-prob, prob])
        
        #print(results)
        return np.array(results)

    # def transform(self, X,):
    #     """Get the distance to each means field.

    #     Parameters
    #     ----------
    #     X : ndarray, shape (n_matrices, n_channels, n_channels)
    #         Set of SPD matrices.

    #     Returns
    #     -------
    #     dist : ndarray, shape (n_matrices, n_classes)
    #         Distance to each means field according to the metric.
    #     """
    #     return self._predict_distances(X)

    # def fit_predict(self, X, y):
    #     """Fit and predict in one function."""
    #     self.fit(X, y)
    #     return self.predict(X)

    # def predict_proba(self, X):
    #     """Predict proba using softmax of negative squared distances.

    #     Parameters
    #     ----------
    #     X : ndarray, shape (n_matrices, n_channels, n_channels)
    #         Set of SPD matrices.

    #     Returns
    #     -------
    #     prob : ndarray, shape (n_matrices, n_classes)
    #         Probabilities for each class.
    #     """
    #     if self.method_label == "lda":
            
    #         dists = self._predict_distances(X)
            
    #         return self.lda.predict_proba(dists)
            
    #     else:
    #         return softmax(-self._predict_distances(X) ** 2)
