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

    def __init__(self, power_list=[-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1], 
                 method_label='lda',
                 metric="riemann",
                 power_mean_zeta = 1e-07, #stopping criterion for the mean (10e-10)
                 distance_squared = True, #squared is better
                 n_jobs=1, 
                 euclidean_mean  = False,
                 distance_strategy = "power_distance",
                 remove_outliers = True,
                 outliers_th = 2.5,
                 outliers_depth = 4, #how many times to run the outliers detection on the same data
                 outliers_max_remove_th = 30,
                 outliers_disable_mean = False,
                 outliers_method = "zscore",
                 outliers_mean_init = True,
                 reuse_previous_mean = False #it is not faster
                 ):
        """Init."""
        self.power_list = power_list
        self.method_label = method_label
        self.metric = metric
        self.n_jobs = n_jobs
        self.euclidean_mean = euclidean_mean #if True sets LogEuclidian distance for LogEuclidian mean and Euclidian distance for power mean p=1
        self.distance_strategy = distance_strategy 
        self.remove_outliers = remove_outliers
        self.outliers_th = outliers_th
        self.outliers_depth = outliers_depth
        self.outliers_max_remove_th = outliers_max_remove_th
        self.outliers_disable_mean = outliers_disable_mean
        self.outliers_method = outliers_method
        self.power_mean_zeta = power_mean_zeta
        self.outliers_mean_init = outliers_mean_init
        self.distance_squared = distance_squared
        self.reuse_previous_mean = reuse_previous_mean
        
        '''
        most used: default_metric and power_distance
        "default_metric" - it uses "metric" (usually Riemann) for all distances 
        "power_mean"     - uses a modified power_distance function based riemann distance, which has an optimization that first calcualtes the inverse of the power mean
        "adaptive1"      - it uses p=-1 harmonic distance and for p=1 euclidean, riemann for the p=0
                         - for the rest it uses custom_distance that uses p to calculate the distance
        "adaptive2"      - it uses p=-1 harmonic distance and for p=1 euclidean, for the rest it uses "metric" (usually Riemann)
        '''
        if distance_strategy not in ["default_metric", "adaptive1", "adaptive2", "power_distance", "custom_distance_function"]:
            raise Exception()("Invalid distance stategy!")
        
        if self.method_label == "lda":
            self.lda = LDA()
    
    #updates the  self.covmeans_ for the provided p for both classes
    def _calculate_mean(self,X, y, p, sample_weight):
        
        means_p = {}
        
        if p == 200: #adding an extra mean - this one is logeuclid and not power mean
            #print("euclidean mean")
            for ll in self.classes_:
                means_p[ll] = mean_logeuclid(
                    X[y == ll],
                    sample_weight=sample_weight[y == ll]
                )
            self.covmeans_[p] = means_p
        else:
            for ll in self.classes_:
                
                init = None
                
                #use previous mean for this p
                #usually when calculating the new mean after outliers removal
                if self.outliers_mean_init and p in self.covmeans_:
                    init = self.covmeans_[p][ll] #use previous mean
                    #print("using init mean")
                
                #use the mean from the previous position in the power list
                elif self.reuse_previous_mean:
                    pos = self.power_list.index(p)
                    if pos>0:
                        prev_p = self.power_list[pos-1]
                        init = self.covmeans_[prev_p][ll]
                        #print(prev_p)
                        #print("using prev mean from the power list")
                    
                means_p[ll] = mean_power_custom(
                    X[y == ll],
                    p,
                    sample_weight=sample_weight[y == ll],
                    zeta = self.power_mean_zeta,
                    init = init
                )
            self.covmeans_[p] = means_p
            
        if self.distance_strategy == "power_distance":
            self.calculate_inv_mean(p)
            
        return means_p #contains means both classes
    
    #removes outliers and calculates the power mean p on the rest
    def _calcualte_mean_remove_outliers(self,X, y, p, sample_weight):
        
        X_no_outliers = X.copy() #so that every power mean p start from the same data
        y_no_outliers = y.copy()
        
        total_outliers_removed_per_class = np.zeros(len(self.classes_))
        total_samples_per_class          = np.zeros(len(self.classes_))
        
        for ll in self.classes_:
            total_samples_per_class[ll] = len(y_no_outliers[y_no_outliers==ll])
            
        is_disabled = False
        
        if self.outliers_method == "iforest":
            iso = IsolationForest(contamination='auto') #0.1
        elif self.outliers_method == "lof":
            lof = LocalOutlierFactor(contamination='auto', n_neighbors=2) #default = 2
        
        for i in range(self.outliers_depth):
            
            if is_disabled and self.outliers_disable_mean:
                break
            
            #print("\nremove outliers iteration: ",i)
            
            #calculate/update the n means (one for each class)
            self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
            
            ouliers_per_iteration_count = {}
            
            #outlier removal is per class
            for ll in self.classes_:
                
                samples_before = X_no_outliers.shape[0]
                
                m = [] #each entry contains a distance to the power mean p for class ll
                
                #length includes all classes, not only the ll
                z_scores = np.zeros(len(y_no_outliers),dtype=float)
            
                # Calcualte all the distances only for class ll and power mean p
                for idx, x in enumerate (X_no_outliers[y_no_outliers==ll]):
                    
                    if self.distance_strategy == "power_distance":
                        dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                    else:
                        dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                    #dist_p = np.log(dist_p)
                    m.append(dist_p)
                
                m = np.array(m, dtype=float)
                
                if self.outliers_method == "zscore":
                    
                    m = np.log(m)
                    # Calculate Z-scores for each data point for the current ll class
                    # For the non ll the zscore stays 0, so they won't be removed
                    z_scores[y_no_outliers==ll] = zscore(m)
                
                    outliers = (z_scores > self.outliers_th) | (z_scores < -self.outliers_th)
                    
                elif self.outliers_method == "iforest":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = iso.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                elif self.outliers_method == "lof":
                    
                    m1 = [[k] for k in m]
                    z_scores[y_no_outliers==ll] = lof.fit_predict(m1)
                    #outliers is designed to be the size with all classes
                    outliers = z_scores == -1
                    
                else:   
                    raise Exception("Invalid Outlier Removal Method")

                outliers_count = len(outliers[outliers==True])
                
                #check if too many samples are about to be removed
                #case 1 less than self.max_outliers_remove_th are to be removed
                if ((total_outliers_removed_per_class[ll] + outliers_count) / total_samples_per_class[ll]) * 100 < self.outliers_max_remove_th:
                    #print ("Removed for class ", ll ," ",  len(outliers[outliers==True]), " samples out of ", X_no_outliers.shape[0])
            
                    X_no_outliers = X_no_outliers[~outliers]
                    y_no_outliers = y_no_outliers[~outliers]
                    sample_weight = sample_weight[~outliers]
                
                    if X_no_outliers.shape[0] != (samples_before - outliers_count):
                        raise Exception("Error while removing outliers!")
                    
                    total_outliers_removed_per_class[ll] = total_outliers_removed_per_class[ll] + outliers_count
                
                else: #case 2 more than self.max_outliers_remove_th are to be removed
                    if self.outliers_disable_mean:
                        is_disabled = True
                        print("WARNING: Power Mean disabled because too many samples were about to be removed for its calculation.")
                        break
                    else:
                        print("WARNING: Skipped full outliers removal because too many samples were about to be removed.")
                
                ouliers_per_iteration_count[ll] = outliers_count
            
            #early stop: if no outliers were removed for both classes then we stop early
            if sum(ouliers_per_iteration_count.values()) == 0:
                break
        
        total_outliers_removed = total_outliers_removed_per_class.sum()
            
        if self.outliers_disable_mean and is_disabled:
            pass #no mean generated (disabled)
            
        elif total_outliers_removed > 0:
           
            #generate the final power mean (after outliers removal)
            self._calculate_mean(X_no_outliers, y_no_outliers, p, sample_weight)
        
            outliers_removed_for_single_mean_gt = X.shape[0] - X_no_outliers.shape[0]
            
            if (total_outliers_removed != outliers_removed_for_single_mean_gt):
                raise Exception("Error outliers removal count!")
            #print("Outliers removed for mean p=",p," is: ",outliers_removed_for_single_mean, " out of ", X.shape[0])
            
            if (outliers_removed_for_single_mean_gt / X.shape[0]) * 100 > self.outliers_max_remove_th:
                raise Exception("Outliers removal algorithm has removed too many samples: ", outliers_removed_for_single_mean_gt, " out of ",X.shape[0])
            
        return is_disabled
    
    def _calculate_all_means(self,X,y,sample_weight):
        
        for p in self.power_list:
            
            if (self.remove_outliers):
                self.covmeans_disabled[p] = self._calcualte_mean_remove_outliers(X, y, p, sample_weight)
            else:
                self._calculate_mean(X, y, p, sample_weight)
                
    def fit(self, X, y, sample_weight=None):
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
        
        if self.euclidean_mean:
            self.power_list.append(200)
            
        self.classes_ = np.unique(y)

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self.covmeans_ = {}
        self.covmeans_disabled = {}
        self.covmeans_inv_ = {}
        
        self._calculate_all_means(X,y,sample_weight)
        
        #start test outliers  init
        # self.remove_outliers = True
        # self.reuse_previous_mean = False
        
        # self.outliers_mean_init = True
        # time_start = perf_counter()
        # self._calculate_all_means(X,y,sample_weight)
        # time_end = perf_counter()
        # time_duration1 = time_end - time_start
        # print("Duration 1:", time_duration1 * 1000)
        
        # self.covmeans_ = {}
        # self.covmeans_disabled = {}
        # self.covmeans_inv_ = {}
        
        # self.outliers_mean_init = False
        # time_start = perf_counter()
        # self._calculate_all_means(X,y,sample_weight)
        # time_end = perf_counter()
        # time_duration2 = time_end - time_start
        # print("Duration 2:", time_duration2 * 1000)
        
        # # report the duration
        # print('Time difference for outliers_mean_init in ms:',(time_duration2 - time_duration1) * 1000)
        #end test
        
        #start test init previous mean
        # self.reuse_previous_mean = True
        # time_start = perf_counter()
        # self._calculate_all_means(X,y,sample_weight)
        # time_end = perf_counter()
        # time_duration1 = time_end - time_start
        # print("Duration 1:", time_duration1 * 1000)
        
        # self.covmeans_ = {}
        # self.covmeans_disabled = {}
        # self.covmeans_inv_ = {}
        
        # self.reuse_previous_mean = False
        # time_start = perf_counter()
        # self._calculate_all_means(X,y,sample_weight)
        # time_end = perf_counter()
        # time_duration2 = time_end - time_start
        # print("Duration 2:", time_duration2 * 1000)
        #end test
        
        # report the duration
        #print('Time difference for reuse_previous_mean in ms:',(time_duration2 - time_duration1) * 1000)
        
        if self.method_label == "lda":
            dists = self._predict_distances(X)
            self.lda.fit(dists,y)

        return self
    
    def calculate_inv_mean(self,p):
              
        means_p = {}
        for ll in self.classes_:
            means_p[ll] = np.linalg.inv(self.covmeans_[p][ll])
        
        self.covmeans_inv_[p] = means_p
            

    def _get_label(self, x, labs_unique):
        
        m = np.zeros((len(self.power_list), len(labs_unique)))
        
        for ip, p in enumerate(self.power_list):
            for ill, ll in enumerate(labs_unique):
                 m[ip, ill] = self._calculate_distance(x,self.covmeans_[p][ll],p)

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
        
        #print("In predict")
        if self.method_label == "lda":

            dists = self._predict_distances(X)
            
            pred  = self.lda.predict(dists)
            
            return np.array(pred)
            
        else:
            
            labs_unique = sorted(self.covmeans_[self.power_list[0]].keys())
    
            pred = Parallel(n_jobs=self.n_jobs)(delayed(self._get_label)(x, labs_unique)
                 for x in X
                )
            
            return np.array(pred)

    def _calculate_distance(self,A,B,p):

        squared = self.distance_squared
        
        if len(A.shape) == 2:
        
            if self.distance_strategy == "adaptive1": #very slow
                
                #                        score       time                     
                #default_metric_r     0.749978   8.547789
                #adaptive1            0.702650  14.123192
                #print("adaptive1")
                
                met = None
                
                if p == 200:
                    met = "logeuclid"
                    #print("euclidean distance")
                    
                if p == 1:
                    met = "euclid"
                
                if p == -1:
                    met = "harmonic"
                
                if p<=0.1 and p>=-0.1:
                    met = "riemann"
                
                if met is None:
                    #print("p based distance")
                    #this is the case for example: -0.75, -0.5, -0.25, +0.75, +0.5, +0.25
                    dist = distance_custom(A, B, k=p, squared = squared)
                    
                    # if (dist is None):
                    #     print("Problem with distance custom")
                else:
                    #when selected one of the above -1,1, [-0.1..0.1]
                    dist = distance(
                            A,
                            B,
                            metric=met,
                            squared = squared,
                        )
                
                #print("adaptive distance", dist)
            
            elif self.distance_strategy == "adaptive2":
                
                #adaptive only for -1,1 and 200
                met = self.metric
                
                if p == 200:
                    met = "logeuclid"
                    #print("euclidean distance")
                    
                elif p == 1:
                    met = "euclid"
                
                elif p == -1:
                    met = "harmonic"
            
                dist = distance(
                        A,
                        B,
                        metric=met,
                        squared = squared,
                    )
                
                #print("adaptive distance", dist)
               
            elif self.distance_strategy == "default_metric":
                
                dist = distance(
                        A,
                        B,
                        metric=self.metric,
                        squared = squared,
                    )
                
            elif self.distance_strategy == "power_distance":
                
                dist = power_distance(
                        A, #trial
                        B, #mean inverted
                        squared = squared,
                    )
                
            elif self.distance_strategy == "custom_distance_function":
                
                dist = distance_custom(A, B, k=p, squared = squared)
                
            else:
                raise Exception("Invalid distance strategy")
                    
        else:
            raise Exception("Error size of input, not matrices?")
            
        return dist
    
    def _calucalte_distances_for_all_means(self,x):
        
        m = {} #contains a distance to a power mean
        
        for p in self.power_list:
            if self.outliers_disable_mean and p in self.covmeans_disabled and self.covmeans_disabled[p]:
                continue
            m[p] = []
            for ll in self.classes_: #add all distances (1 per class) for m[p] power mean
                if self.distance_strategy == "power_distance":
                    
                    #time_start = perf_counter_ns()
                    dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
                    #time_end = perf_counter_ns()
                    #time_duration = time_end - time_start
                    # report the duration
                    #print(f'Took {time_duration} seconds inverse distance')
                    
                    #test code (to be removed)
                    # self.distance_strategy = "default_metric"
                    # time_start = perf_counter_ns()
                    # dist_old = self._calculate_distance(x, self.covmeans_[p][ll], p)
                    # time_end = perf_counter_ns()
                    # self.distance_strategy = "power_distance"
                    # time_duration = time_end - time_start
                    # # report the duration
                    # print(f'Took {time_duration} seconds normal distance')
                    # print(dist_p, dist_old, round(dist_p - dist_old,4))
                    # if abs(dist_p - dist_old > 0.001):
                    #     raise Exception("Error distance")
                    #end test code

                else:
                    dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
                m[p].append(dist_p)
                
        combined = [] #combined for all classes
        for v in m.values():
            combined.extend(v)
        
        combined = np.array(combined)
        
        return combined
        
    def _predict_distances(self, X):
        """Helper to predict the distance. Equivalent to transform."""
        
        #print("predict distances")
        
        # if self.ts_enabled:
            
        #     self.ts_covmeans_ = copy.deepcopy(self.covmeans_)
            
        #     for p in self.power_list:
        #         for ll in self.classes_:
        #             #print(np.array([self.covmeans_[p][ll]]).shape)
        #             self.ts_covmeans_[p][ll] = self.ts.transform(np.array([self.covmeans_[p][ll]]))
            
        #     X = self.ts.transform(X)
           
        if (self.n_jobs == 1):
            distances = []
            for x in X:
                distances_per_mean = self._calucalte_distances_for_all_means(x)
                distances.append(distances_per_mean)
        else:
            distances = Parallel(n_jobs=self.n_jobs)(delayed(self._calucalte_distances_for_all_means)(x)
                 for x in X
                )
            
            distances = np.array(distances)
        
        return distances

    def transform(self, X,):
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
        if self.method_label == "lda":
            
            dists = self._predict_distances(X)
            
            return self.lda.predict_proba(dists)
            
        else:
            return softmax(-self._predict_distances(X) ** 2)
