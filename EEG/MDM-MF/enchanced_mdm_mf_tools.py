# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:08:36 2024

@author: antona
"""

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.spatialfilters import CSP
from pyriemann.utils.ajd import rjd, ajd_pham, ajd, uwedge
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize

from pyriemann.utils.ajd import ajd_pham
from pyriemann.utils.base import sqrtm, invsqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann, distance_euclid
from pyriemann.utils.geodesic import geodesic_riemann
from pyriemann.utils.utils import check_weights, check_function
from pyriemann.utils.mean import mean_euclid, mean_riemann, mean_harmonic #_deprecate_covmats
import warnings
import scipy
from sklearn import decomposition
from pyriemann.utils.distance import _check_inputs
import math
from scipy.stats import zscore

#requires a version of pyRiemann where CSP accepts maxiter and n_iter_max
# class CustomCspTransformer(BaseEstimator, TransformerMixin):
    
#     def __init__(self,mode = 0):
        
#         '''
#         0 fastest
#         1 slower, better performance
#         2 slow, better performance
#         3 slowest
#         '''
        
#         self.mode = mode
#         self.nfilter = 10
        
#         if mode == 0:
#             self.maxiter = 5
#             self.n_iter_max = 5
#         elif mode == 1:
#             self.maxiter = 30
#             self.n_iter_max = 30
#         elif mode == 2:
#             self.maxiter = 50
#             self.n_iter_max = 100
#         elif mode == 3:
#             self.maxiter = 50
#             self.n_iter_max = 100
    
#     def fit(self, X, y=None):
        
#         self.n_electrodes = X.shape[1]
        
#         if self.n_electrodes <= self.nfilter:
#             self.csp = CSP(metric="euclid", log=False)
#             self.csp.fit(X,y)
#             return self
        
#         elif self.n_electrodes > 32:
            
#             self.csp1 = CSP(nfilter = 32, metric="euclid", log=False)
#             self.csp2 = CSP(metric = "ale", nfilter = self.nfilter, log=False, maxiter = self.maxiter, n_iter_max = self.n_iter_max) # maxiter = 50, n_iter_max = 100)
            
#         else: #<=32
            
#             if self.mode == 3:
#                 self.csp = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = self.maxiter, n_iter_max = self.n_iter_max) # maxiter = 50, n_iter_max = 100)
#                 self.csp.fit(X,y)
#                 return self
#             else:
#                 self.csp1 = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
#                 self.csp2 = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = self.maxiter, n_iter_max = self.n_iter_max) # maxiter = 50, n_iter_max = 100)
            
#         self.csp1.fit(X,y)
#         X1 = self.csp1.transform(X)
#         self.csp2.fit(X1,y)
        
#         return self
    
#     def transform(self, X):
        
#         if self.n_electrodes <= self.nfilter or (self.mode == 3 and self.n_electrodes <= 32):
            
#             X_transformed = self.csp.transform(X)
#             return X_transformed
        
#         else:
            
#             X_transformed1 = self.csp1.transform(X)
#             X_transformed2 = self.csp2.transform(X_transformed1)
#             return X_transformed2

class CustomCspTransformer2(BaseEstimator, TransformerMixin):
    
    def __init__(self, mode, nfilter = None):
        
        self.mode    = mode
        self.nfilter = nfilter #default 10
        self.nfilter_dimensionality_reduction = 28 #threshold between the two modes
    
    def fit(self, X, y=None):
          
        self.n_electrodes = X.shape[1]
        
        if self.nfilter is None:
            #self.nfilter = max( int(math.sqrt(self.n_electrodes)) * 2, 4)
            #self.nfilter = int(math.sqrt(self.n_electrodes)) * 2
            self.nfilter = 10 #default value
        
        #print("nfilter: ",self.nfilter)
        
        if self.n_electrodes <= self.nfilter:
            #print("not processed")
            return self
            # self.nfilter = self.n_electrodes
            # print("self.n_electrodes <= self.nfilter")
            # self.csp = CSP(nfilter = self.n_electrodes)
                           
        elif self.mode == "high_electrodes_count":
            if self.n_electrodes > self.nfilter_dimensionality_reduction:
                self.csp = CSP(nfilter = self.nfilter_dimensionality_reduction, metric="euclid", log=False)
            else:
                return self
            
        elif self.mode == "low_electrodes_count":
            
            if self.n_electrodes > self.nfilter_dimensionality_reduction:
                raise Exception("Number of electrodes too high. CSP will be slow. Use 'pre-rocessing' mode instead.")
            else: # <28 electrodes 
                self.csp = CSP(nfilter = self.nfilter, metric = "riemann", log=False)
        else:
            raise Exception("Invalid CSP mode")
             
        self.csp.fit(X,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter:
            #P = self.csp.transform(X)
            return X
        
        if self.mode == "high_electrodes_count":
            
            if self.n_electrodes > self.nfilter_dimensionality_reduction:
                return self.csp.transform(X)
            else: 
                return X
            
        elif self.mode == "low_electrodes_count":
            
            if self.n_electrodes > self.nfilter_dimensionality_reduction:
                raise Exception("Number of electrodes too high. CSP will be slow. Use in 'pre-rocessing' mode first.")
            
            else:
                return self.csp.transform(X)
        else:
             raise Exception("Invalid CSP mode")
             
# class CustomCspTransformer3(BaseEstimator, TransformerMixin):
    
#     def __init__(self, mode = "fast_dimensionality_reduction", speed = 3):
        
#         self.mode = mode
#         self.nfilter = 10
#         self.speed = speed
        
#         if speed == 0:
#             self.maxiter = 5
#             self.n_iter_max = 5
#         elif speed == 1:
#             self.maxiter = 30
#             self.n_iter_max = 30
#         elif speed == 2:
#             self.maxiter = 50
#             self.n_iter_max = 100
#         elif speed == 3:
#             self.maxiter = 50
#             self.n_iter_max = 100
    
#     def fit(self, X, y=None):
        
#         self.n_electrodes = X.shape[1]
        
#         if self.n_electrodes <= self.nfilter:
#             return self
        
#         if self.mode == "fast_dimensionality_reduction":
            
#             if self.n_electrodes > 28:
#                 self.csp = CSP(nfilter = 28, metric="euclid", log=False)
#             else:
#                 return self
            
#         elif self.mode == "slow_ale":
            
#             if self.n_electrodes > 28:
#                 raise Exception("Number of electrodes too high. CSP will be slow. Use 'pre-rocessing' mode instead.")
#             else: # <28 electrodes 
#                 self.csp = CSP(nfilter = self.nfilter, metric = "ale", log=False, maxiter = self.maxiter, n_iter_max = self.n_iter_max) # maxiter = 50, n_iter_max = 100)
#         else:
#             raise Exception("Invalid mode")
             
#         self.csp.fit(X,y)
        
#         return self
    
#     def transform(self, X):
        
#         if self.n_electrodes <= self.nfilter:
#             return X
        
#         if self.mode == "fast_dimensionality_reduction":
            
#             if self.n_electrodes > 28:
#                 return self.csp.transform(X)
#             else: 
#                 return X
            
#         elif self.mode == "slow_ale":
            
#             if self.n_electrodes > 28:
#                 raise Exception("Number of electrodes too high. CSP will be slow. Use in 'pre-rocessing' mode first.")
            
#             else:
#                 return self.csp.transform(X)
#         else:
#              raise Exception("Invalid mode")

class Diagonalizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, diag_method="rjd", norm_method="L2", n_iter_max = 200):
        self.diag_method = diag_method
        self.norm_method = norm_method
        self.n_iter_max = n_iter_max
        
    def fit(self, X, y=None):
        
        #print("fit strict diagonalizer")
        
        #uwedge, rjd, ajd, 
        if self.diag_method == "rjd":
            
            self.V = rjd(X, n_iter_max=self.n_iter_max) #uses a modified version where D is not calculated to save time
            #self.V = rjd(X, n_iter_max=500)
            #self.V = uwedge(X, n_iter_max=500)
            #self.V = ajd(X, n_iter_max=500)
           
        elif self.diag_method == "ajd_pham":
            
            V,_ = ajd_pham(X, n_iter_max=self.n_iter_max) #200 is good
            
            V = V.transpose()
            
            if self.norm_method == "L2":
                
                V = normalize(V, norm="l2", axis = 0)
                        
                self.V = V
                
                #verify V
                diff = 1 - norm(self.V, axis=0)
                if all(np.absolute(diff) > 0.0001):
                    raise Exception("Columns not converted to L2 norm! Difference is: ", diff)
                
            elif self.norm_method == "mean_std":
                
                col_means = np.mean(V, axis=0)
    
                # normalize each column by subtracting its mean and dividing by its standard deviation
                arr_normalized = (V - col_means) / np.std(V, axis=0)
                
                self.V = arr_normalized
            else:
                raise Exception("Incorrect norm method!")
        else:
            raise Exception("Incorrect diagonalization method!")
            
        return self
        
    def reconstruct_covariance(self, X):
        
        dominantly_diagonal = np.matmul(np.matmul(self.V.transpose() , X) , self.V)
        
        striclty_diagonal = np.diag(np.diag(dominantly_diagonal))
        
        reconstructed_covariance = np.matmul(np.matmul(self.V , striclty_diagonal) , self.V.transpose())
        
        return reconstructed_covariance
    
    def transform(self, X):
        
        return np.array([self.reconstruct_covariance(xi) for xi in X])
    
class PCA_SPD(TransformerMixin):
    
    def __init__(self, n_components):
        self.n_components = n_components
    
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

    def fit(self, X, y):
        return self
    
    def transform(self, X, y=None):
        
        X = np.array(X)
        new_dataset = np.zeros((X.shape[0], self.n_components, self.n_components))
        
        for i in range(X.shape[0]):
            
            covariance_to_reduce = X[i]
            
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_to_reduce) # Calculate eigenvalues and eigenvectors            
            
            idx = eigenvalues.argsort()[::-1] # Sort eigenvalues in descending order
            
            eigenvectors = eigenvectors[:,idx][:,:self.n_components] # Sort eigenvectors according to eigenvalues and get the first n_components
            
            reduced_covariance = eigenvectors.T @ np.diag(eigenvalues) @ eigenvectors            
            
            new_dataset[i] = reduced_covariance
            
        return new_dataset 

#this is a custom distance that gets an additional parameter
def distance_custom(A, B, k, squared=False):
    r"""Harmonic distance between invertible matrices.

    The harmonic distance between two invertible matrices :math:`\mathbf{A}`
    and :math:`\mathbf{B}` is:

    .. math::
        d(\mathbf{A},\mathbf{B}) =
        \Vert \mathbf{A}^{-1} - \mathbf{B}^{-1} \Vert_F

    Parameters
    ----------
    A : ndarray, shape (..., n, n)
        First invertible matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, n)
        Second invertible matrices, same dimensions as A.
    squared : bool, default False
        Return squared distance.

        .. versionadded:: 0.5

    Returns
    -------
    d : float or ndarray, shape (...,)
        Harmonic distance between A and B.

    See Also
    --------
    distance
    """
    
    #return distance_euclid(scipy.linalg.fractional_matrix_power(A,k), scipy.linalg.fractional_matrix_power(B,k), squared=squared)
    A1 = scipy.linalg.fractional_matrix_power(A,k)
    B1 = scipy.linalg.fractional_matrix_power(B,k)
    
    # if (A is None):
    #     print("A is none in distance_custom")
        
    # if (B is None):
    #     print("B is none in distance_custom")
    
    dist = distance_euclid(A1, B1, squared=squared)
    
    return dist

def power_distance(trial, power_mean_inv, squared=False):
    '''
    RMF works better with squared=False

    Parameters
    ----------
    trial : TYPE
        DESCRIPTION.
    power_mean_inv : TYPE
        DESCRIPTION.
    squared : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    # _check_inputs(A, B)
    #d2 = (np.log(_recursive(eigvalsh, A, B))**2).sum(axis=-1)
    
    # return d2 if squared else np.sqrt(d2)
    _check_inputs(power_mean_inv, trial)
    #scipy.linalg.eigvals (general, non symetric)
    #np.linalg.eigvals    (general, non symetric)
    #but power_mean_inv @ trial is not symetric, scipy.linalg.eigvals is more correct
    #eigvals gives a vector
    #.sum(axis=-1) = sum()
    #with np.log and **2 it is still a 1D vector
    d2 = (np.log( np.linalg.eigvals (power_mean_inv @ trial)) **2 ).sum(axis=-1)
    #same as d2 = (np.power(np.log( np.linalg.eigvals(power_mean_inv @ trial)),2)).sum()
    
    #scipy.linalg.eigvals with real() is slower than numpy
    #d2 = (np.log( np.real(  scipy.linalg.eigvals (power_mean_inv @ trial, check_finite = False) ) ) **2 ).sum(axis=-1)
    #print("power distance")
    #return d2 ** 2 if squared else d2
    
    #correct version:
    return d2 if squared else np.sqrt(d2)
    #ADD an extra paramter for the function that does ln(d2) if true after the squared distance above

def vector_distance(trial, power_mean_inv, vector_distance_method = 2):
    '''
    The distance output is a vector instead of a single number.
    '''
    _check_inputs(power_mean_inv, trial)
    
    ev = np.linalg.eigvals (power_mean_inv @ trial)
    s = (np.log(ev) ** 2).sum(axis=-1)
    #s = ev.sum(axis=-1)
    
    #ev2 = ev[0:10] #very good
    #ev2 = np.append(ev[0:10],ev[18:22])

    n = trial.shape[0]
    #print(n)
    #ev2 = ev[int(n/3):2 * int(n/3) ] #get the middle 1/3
    #ev2 = ev[int(n/3):int(n/3) + 2 ] 
    #ev2 = ev[1::2] #good
    #ev2 = ev[0:int(len(ev2)/2)]
    ev2 = ev[0:3] 
    ev3 = ev[0:4]
    ev4 = ev[0:6]
    
    
    #print(vector_distance_method)
    #1) ln only
    if vector_distance_method == 1:
        return np.log(ev2)

    #2) ln and ** 2
    if vector_distance_method == 2:
        #return (np.log(ev2) ** 2).append((np.log(ev) ** 2).sum(axis=-1))
        #r = np.append(np.ones(1),(np.log(ev) ** 2).sum(axis=-1))
        
        #best 0.85
        #r = np.append(ev,s) # 0.84
        #r = ev #0.73
        #r = np.append(ev2,s) # 0.84 ev[-5]. 0.85
        
        #good
        r = np.array(1)
        r = np.append( r, s )
        #r = np.append( r,1 )
        #r = np.append(r, (np.log(ev2) ** 2).sum(axis=-1))
        #r = np.append(r, (np.log(ev3) ** 2).sum(axis=-1))
        #r = np.append(r, ev2)
        
        #r = np.append(r,(np.log(ev2) ** 2).sum(axis=-1)) # 0.84 ev[0:10]. 0.845
        #r = np.append(np.log(ev2) ** 2,s) # ev[0:10] 
        #print(r)
        return r
    
    #3) directly eigvalues, so neither ** 2, neither ln
    if vector_distance_method == 3:
        return ev2

def mean_power_custom(X=None, p=None, *, init=None, sample_weight=None, zeta=10e-10, maxiter=150, #default = 100
               covmats=None):
    r"""Power mean of SPD/HPD matrices.

    Power mean of order p is the solution of [1]_ [2]_:

    .. math::
        \mathbf{M} = \sum_i w_i \ \mathbf{M} \sharp_p \mathbf{X}_i

    where :math:`\mathbf{A} \sharp_p \mathbf{B}` is the geodesic between
    matrices :math:`\mathbf{A}` and :math:`\mathbf{B}`.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    p : float
        Exponent, in [-1,+1]. For p=0, it returns
        :func:`pyriemann.utils.mean.mean_riemann`.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    zeta : float, default=10e-10
        Stopping criterion.
    maxiter : int, default=100
        The maximum number of iterations.

    Returns
    -------
    M : ndarray, shape (n, n)
        Power mean.

    Notes
    -----
    .. versionadded:: 0.3

    See Also
    --------
    mean_covariance

    References
    ----------
    .. [1] `Matrix Power means and the Karcher mean
        <https://www.sciencedirect.com/science/article/pii/S0022123611004101>`_
        Y. Lim and M. Palfia. Journal of Functional Analysis, Volume 262,
        Issue 4, 15 February 2012, Pages 1498-1514.
    .. [2] `Fixed Point Algorithms for Estimating Power Means of Positive
        Definite Matrices
        <https://hal.archives-ouvertes.fr/hal-01500514>`_
        M. Congedo, A. Barachant, and R. Bhatia. IEEE Transactions on Signal
        Processing, Volume 65, Issue 9, pp.2211-2220, May 2017
    """
    #X = _deprecate_covmats(covmats, X)
    if p is None:
        raise ValueError("Input p can not be None")
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p < -1 or 1 < p:
        raise ValueError("Exponent must be in [-1,+1]")

    if p == 1:
        return mean_euclid(X, sample_weight=sample_weight)#,0)
    #elif p == 0:
    elif p == 0:# or (p < 0.01 and p > -0.01): #Anton1: added (p < 0.01 and p>-0.01) for when p=0.001 instead of 0
        return mean_riemann(X, 
                            sample_weight = sample_weight, 
                            init          = init,   #Anton2: added init, now in pyRiemann
                            tol           = zeta,   #Anton3: added zeta here decreases the number significant digits
                            maxiter       = maxiter #increased from default 50 to 100
                            )
    elif p == -1:
        return mean_harmonic(X, sample_weight=sample_weight)#,0)

    n_matrices, n, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    phi = 0.375 / np.abs(p)

    #Anton4: added init, there was no support for init before for the below calculation        
    if init is None:
        G = powm(np.einsum("a,abc->bc", sample_weight, powm(X, p)), 1/p) #with bug fix
    else:
        G = init
        
    if p > 0:
      K = invsqrtm(G)
    else:
      K = sqrtm(G)

    eye_n, sqrt_n = np.eye(n), np.sqrt(n)
    crit = 10 * zeta
    
    #itr = 0
    
    for _ in range(maxiter):
        H = np.einsum(
            'a,abc->bc',
            sample_weight,
            powm(K @ powm(X, np.sign(p)) @ K.conj().T, np.abs(p))
        )
        K = powm(H, -phi) @ K

        crit = np.linalg.norm(H - eye_n) / sqrt_n
        if crit <= zeta:
            break
        
        #itr = itr + 1
    else:
        warnings.warn("Power mean convergence not reached")

    M = K.conj().T @ K
    if p > 0:
        M = np.linalg.inv(M)

    return M#, itr

# def mean_remove_outliers(X, 
#                           y, 
#                           sample_weight, 
#                           mean_func, 
#                           dist_func, 
#                           method="zscore",
#                           outliers_th = 2.5,
#                           outliers_depth = 4, #how many times to run the outliers detection on the same data
#                           outliers_max_remove_th = 30,
#                           **kwargs
#                           ):
    
#     classes = np.unique(y)

#     if sample_weight is None:
#         sample_weight = np.ones(X.shape[0])
        
#     X_no_outliers = X.copy() #so that every power mean p start from the same data
#     y_no_outliers = y.copy()
    
#     total_outliers_removed_per_class = np.zeros(len(classes))
#     total_samples_per_class          = np.zeros(len(classes))
    
#     for ll in classes:
#         total_samples_per_class[ll] = len(y_no_outliers[y_no_outliers==ll])
    
#     for i in range(outliers_depth):
        
#         #print("\nremove outliers iteration: ",i)
        
#         #calculate/update the n means (one for each class)
#         current_mean = mean_func(X_no_outliers, y_no_outliers, kwargs) #p, sample_weight
        
#         ouliers_per_iteration_count = {}
        
#         #outlier removal is per class
#         for ll in classes:
            
#             samples_before = X_no_outliers.shape[0]
            
#             m = [] #each entry contains a distance to the power mean p for class ll
            
#             #length includes all classes, not only the ll
#             z_scores = np.zeros(len(y_no_outliers),dtype=float)
        
#             # Calculate all the distances only for class ll and power mean p
#             for idx, x in enumerate (X_no_outliers[y_no_outliers==ll]):
                
#                 dist_p = dist_func(x, current_mean, kwargs)
                
#                 # if self.distance_strategy == "power_distance":
#                 #     dist_p = self._calculate_distance(x, self.covmeans_inv_[p][ll], p)
#                 # else:
#                 #     dist_p = self._calculate_distance(x, self.covmeans_[p][ll], p)
#                 m.append(dist_p)
            
#             m = np.array(m, dtype=float)
            
#             if method == "zscore":
                
#                 m = np.log(m)
#                 # Calculate Z-scores for each data point for the current ll class
#                 # For the non ll the zscore stays 0, so they won't be removed
#                 z_scores[y_no_outliers==ll] = zscore(m)
            
#                 outliers = (z_scores > outliers_th) | (z_scores < -outliers_th)
                
#             # elif self.outliers_method == "iforest":
                
#             #     m1 = [[k] for k in m]
#             #     z_scores[y_no_outliers==ll] = iso.fit_predict(m1)
#             #     #outliers is designed to be the size with all classes
#             #     outliers = z_scores == -1
                
#             # elif self.outliers_method == "lof":
                
#             #     m1 = [[k] for k in m]
#             #     z_scores[y_no_outliers==ll] = lof.fit_predict(m1)
#             #     #outliers is designed to be the size with all classes
#             #     outliers = z_scores == -1
                
#             else:   
#                 raise Exception("Invalid Outlier Removal Method")

#             outliers_count = len(outliers[outliers==True])
            
#             #check if too many samples are about to be removed
#             #case 1 less than self.max_outliers_remove_th are to be removed
#             if ((total_outliers_removed_per_class[ll] + outliers_count) / total_samples_per_class[ll]) * 100 < outliers_max_remove_th:
#                 #print ("Removed for class ", ll ," ",  len(outliers[outliers==True]), " samples out of ", X_no_outliers.shape[0])
        
#                 X_no_outliers = X_no_outliers[~outliers]
#                 y_no_outliers = y_no_outliers[~outliers]
#                 sample_weight = sample_weight[~outliers]
            
#                 if X_no_outliers.shape[0] != (samples_before - outliers_count):
#                     raise Exception("Error while removing outliers!")
                
#                 total_outliers_removed_per_class[ll] = total_outliers_removed_per_class[ll] + outliers_count
            
#             else: #case 2 more than self.max_outliers_remove_th are to be removed
#                 # if self.outliers_disable_mean:
#                 #     is_disabled = True
#                 #     print("WARNING: Power Mean disabled because too many samples were about to be removed for its calculation.")
#                 #     break
#                 # else:
#                 print("WARNING: Skipped full outliers removal because too many samples were about to be removed.")
            
#             ouliers_per_iteration_count[ll] = outliers_count
        
#         #early stop: if no outliers were removed for both classes then we stop early
#         if sum(ouliers_per_iteration_count.values()) == 0:
#             break
    
#     total_outliers_removed = total_outliers_removed_per_class.sum()
        
#     # if outliers_disable_mean and is_disabled:
#     #     pass #no mean generated (disabled)
        
#     #elif total_outliers_removed > 0:
#     if total_outliers_removed > 0:
       
#         #generate the final power mean (after outliers removal)
#         current_mean = mean_func(X_no_outliers, y_no_outliers, kwargs)
    
#         outliers_removed_for_single_mean_gt = X.shape[0] - X_no_outliers.shape[0]
        
#         if (total_outliers_removed != outliers_removed_for_single_mean_gt):
#             raise Exception("Error outliers removal count!")
#         #print("Outliers removed for mean p=",p," is: ",outliers_removed_for_single_mean, " out of ", X.shape[0])
        
#         if (outliers_removed_for_single_mean_gt / X.shape[0]) * 100 > outliers_max_remove_th:
#             raise Exception("Outliers removal algorithm has removed too many samples: ", outliers_removed_for_single_mean_gt, " out of ",X.shape[0])
            
#     return current_mean
    