'''

Contains a modified version of MeanField

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
from pyriemann.utils.mean import mean_covariance, mean_power
from pyriemann.utils.distance import distance
from pyriemann.tangentspace import FGDA, TangentSpace


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

class MeanField(BaseEstimator, ClassifierMixin, TransformerMixin):
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

    def __init__(self, power_list=[-1, 0, 1], method_label='sum_means',
                 metric="riemann", n_jobs=1):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

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
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.covmeans_ = {}
        for p in self.power_list:
            means_p = {}
            for ll in self.classes_:
                means_p[ll] = mean_power(
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll]
                )
            self.covmeans_[p] = means_p

        return self

    def _get_label(self, x, labs_unique):
        
        m = np.zeros((len(self.power_list), len(labs_unique)))
        
        for ip, p in enumerate(self.power_list):
            for ill, ll in enumerate(labs_unique):
                m[ip, ill] = distance(
                    x, self.covmeans_[p][ll], metric=self.metric, squared=True)

        if self.method_label == 'sum_means':
            ipmin = np.argmin(np.sum(m, axis=1))
        elif self.method_label == 'inf_means':
            ipmin = np.where(m == np.min(m))[0][0]
        else:
            raise TypeError('method_label must be sum_means or inf_means')

        y = labs_unique[np.argmin(m[ipmin])]
        return y

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
        labs_unique = sorted(self.covmeans_[self.power_list[0]].keys())

        pred = Parallel(n_jobs=self.n_jobs)(delayed(self._get_label)(x, labs_unique)
             for x in X
            )
        
        return np.array(pred)

    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""

        dist = []
        for x in X:
            m = {} #m contains a distance to a power mean
            
            #TODO: needs to be done parallel
            for p in self.power_list:
                m[p] = []
                for ll in self.classes_: #add all distances (1 per class) for m[p] power mean
                    m[p].append(
                        distance(
                            x,
                            self.covmeans_[p][ll],
                            metric=self.metric,
                        )
                    )
                    
            combined = []
            for v in m.values():
                combined.extend(v)
            
            combined = np.array(combined)
            
            #print("Combined: ", len(self.power_list) , len(self.classes_), combined.shape)
            dist.append(combined)
        
        dist = np.array(dist)
        #print("Dist shape", dist.shape)
        return dist

    def transform(self, X):
        """Get the distance to each means field.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_matrices, n_classes)
            Distance to each means field according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax of negative squared distances.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_matrices, n_classes)
            Probabilities for each class.
        """
        return softmax(-self._predict_distances(X) ** 2)
