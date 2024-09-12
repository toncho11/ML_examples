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
from pyriemann.utils.mean import mean_euclid, mean_riemann, mean_harmonic, _deprecate_covmats
import warnings
import scipy
from sklearn import decomposition

#requires a version of pyRiemann where CSP accepts maxiter and n_iter_max
class CustomCspTransformer(BaseEstimator, TransformerMixin):
    
    #for 128 electrodes ale is better than euclid but slower
    # Averaging the session performance:
    #                       score       time
    # pipeline                              
    # CSP_A_PM12_LDA_CD  0.853723  16.892366
    # CSP_E_PM12_LDA_CD  0.823720   1.168123
    # TSLR               0.875154   2.505664
    # Difference is with 1 star confidence in favor of ALE
    def __init__(self, nfilter = 10): #maxiter_128 = 2, n_iter_max_128=2):
        #self.euclid = euclid
        self.nfilter = nfilter
        #self.maxiter_128 = maxiter_128
        #self.n_iter_max_128 = n_iter_max_128
    
    def fit(self, X, y=None):
        #print("fit csp")
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:  #do nothing
            return self
        
        elif self.n_electrodes <= 64: #default < 60
            
            #n_iter_max=200 needs to be set in the code for "ale"
            #good maxiter = 20, n_iter_max = 10
            #5 5 ok
            
            #self.nfilter = 6
            self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 10, n_iter_max = 8) # maxiter = 50, n_iter_max = 100)
        
        else: #n_electrodes > 64
            # if self.euclid:
            #     self.csp = CSP(metric = "euclid", nfilter=self.nfilter, log=False) #pca par example
            # else:
            #self.nfilter = 10
            self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 2 , n_iter_max = 2) #pca par example
            
        self.csp.fit(X,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter: #return unprocessed data
            return X 
        else:
            X_transformed = self.csp.transform(X)
            return X_transformed

class CustomCspTransformer2(BaseEstimator, TransformerMixin):
    
    #for 128 electrodes ale is better than euclid but slower
    # Averaging the session performance:
    #                       score       time
    # pipeline                              
    # CSP_A_PM12_LDA_CD  0.853723  16.892366
    # CSP_E_PM12_LDA_CD  0.823720   1.168123
    # TSLR               0.875154   2.505664
    # Difference is with 1 star confidence in favor of ALE
    def __init__(self, nfilter = 10): #maxiter_128 = 2, n_iter_max_128=2):
        #self.euclid = euclid
        self.nfilter = nfilter
        #self.maxiter_128 = maxiter_128
        #self.n_iter_max_128 = n_iter_max_128
    
    def fit(self, X, y=None):
        #print("fit csp")
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            
            self.csp = CSP(nfilter = self.nfilter,log=False)
        
        elif self.n_electrodes <= 64: #default < 60
            
            #n_iter_max=200 needs to be set in the code for "ale"
            #good maxiter = 20, n_iter_max = 10
            #5 5 ok
            
            #self.nfilter = 6
            self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 10, n_iter_max = 8) # maxiter = 50, n_iter_max = 100)
        
        else: #n_electrodes > 64
            # if self.euclid:
            #     self.csp = CSP(metric = "euclid", nfilter=self.nfilter, log=False) #pca par example
            # else:
            #self.nfilter = 10
            self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 2 , n_iter_max = 2) #pca par example
            
        self.csp.fit(X,y)
        
        return self
    
    def transform(self, X):
        
        # if self.n_electrodes <= self.nfilter: #return unprocessed data
        #     return X 
        # else:
            X_transformed = self.csp.transform(X)
            return X_transformed

class CustomCspTransformer3(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nfilter = 10
    
    def fit(self, X, y=None):
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            self.csp = CSP(metric="euclid", log=False)
            self.csp.fit(X,y)
            return self
        
        elif self.n_electrodes > 32:
            
            self.csp1 = CSP(nfilter = 32, metric="euclid", log=False)
            self.csp2 = CSP(metric = "ale", nfilter = self.nfilter, log=False, maxiter = 2, n_iter_max = 2) # maxiter = 50, n_iter_max = 100)
            
        else: #<=32
            
            self.csp1 = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
            self.csp2 = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = 2, n_iter_max = 2) # maxiter = 50, n_iter_max = 100)
            
        self.csp1.fit(X,y)
        X1 = self.csp1.transform(X)
        self.csp2.fit(X1,y)
        
        return self

class CustomCspTransformer4(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nfilter = 10
    
    def fit(self, X, y=None):
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            self.csp = CSP(metric="euclid", log=False)
            self.csp.fit(X,y)
            return self
        
        elif self.n_electrodes > 32:
            
            self.csp1 = CSP(nfilter = 32, metric="euclid", log=False)
            self.csp2 = CSP(metric = "ale", nfilter = self.nfilter, log=False, maxiter = 30, n_iter_max = 30) # maxiter = 50, n_iter_max = 100)
            
        else: #<=32
            
            self.csp1 = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
            self.csp2 = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = 30, n_iter_max = 30) # maxiter = 50, n_iter_max = 100)
            
        self.csp1.fit(X,y)
        X1 = self.csp1.transform(X)
        self.csp2.fit(X1,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter:
            
            X_transformed = self.csp.transform(X)
            return X_transformed
        
        else:
            
            X_transformed1 = self.csp1.transform(X)
            X_transformed2 = self.csp2.transform(X_transformed1)
            return X_transformed2

class CustomCspTransformer5(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nfilter = 10
    
    def fit(self, X, y=None):
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            self.csp = CSP(metric="euclid", log=False)
            self.csp.fit(X,y)
            return self
        
        elif self.n_electrodes > 32:
            
            self.csp1 = CSP(nfilter = 32, metric="euclid", log=False)
            self.csp2 = CSP(metric = "ale", nfilter = self.nfilter, log=False, maxiter = 50, n_iter_max = 100) # maxiter = 50, n_iter_max = 100)
            
        else: #<=32
            
            self.csp1 = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
            self.csp2 = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = 50, n_iter_max = 100) # maxiter = 50, n_iter_max = 100)
            
        self.csp1.fit(X,y)
        X1 = self.csp1.transform(X)
        self.csp2.fit(X1,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter:
            
            X_transformed = self.csp.transform(X)
            return X_transformed
        
        else:
            
            X_transformed1 = self.csp1.transform(X)
            X_transformed2 = self.csp2.transform(X_transformed1)
            return X_transformed2

class CustomCspTransformer6(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nfilter = 10
    
    def fit(self, X, y=None):
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            self.csp = CSP(metric="euclid", log=False)
            self.csp.fit(X,y)
            return self
        
        else:
            
            self.csp1 = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
            #self.csp2 = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = 50, n_iter_max = 100) # maxiter = 50, n_iter_max = 100)
            
        self.csp1.fit(X,y)
        #X1 = self.csp1.transform(X)
        #self.csp2.fit(X1,y)
        
        return self
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter:
            
            X_transformed = self.csp.transform(X)
            return X_transformed
        
        else:
            
            #X_transformed1 = self.csp1.transform(X)
            X_transformed2 = self.csp1.transform(X)
            return X_transformed2

class CustomCspTransformer7(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        self.nfilter = 10
    
    def fit(self, X, y=None):
        
        self.n_electrodes = X.shape[1]
        
        if self.n_electrodes <= self.nfilter:
            self.csp = CSP(metric="euclid", log=False)
            self.csp.fit(X,y)
            return self
        
        elif self.n_electrodes > 32:
            
            self.csp1 = CSP(nfilter = 32, metric="euclid", log=False)
            self.csp2 = CSP(metric = "ale", nfilter = self.nfilter, log=False, maxiter = 50, n_iter_max = 100) # maxiter = 50, n_iter_max = 100)
            
            self.csp1.fit(X,y)
            X1 = self.csp1.transform(X)
            self.csp2.fit(X1,y)
            
            return self
        
        else: #<=32
            
            #self.csp = CSP(nfilter = self.nfilter, metric ="euclid", log=False)
            self.csp = CSP(nfilter = self.nfilter, metric = "ale"  , log=False, maxiter = 50, n_iter_max = 100) # maxiter = 50, n_iter_max = 100)
            self.csp.fit(X,y)
            return self
        
    
    def transform(self, X):
        
        if self.n_electrodes <= self.nfilter or self.n_electrodes <=32:
            
            X_transformed = self.csp.transform(X)
            return X_transformed
        
        else:
            
            X_transformed1 = self.csp1.transform(X)
            X_transformed2 = self.csp2.transform(X_transformed1)
            return X_transformed2

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
    X = _deprecate_covmats(covmats, X)
    if p is None:
        raise ValueError("Input p can not be None")
    if not isinstance(p, (int, float)):
        raise ValueError("Power mean only defined for a scalar exponent")
    if p < -1 or 1 < p:
        raise ValueError("Exponent must be in [-1,+1]")

    if p == 1:
        return mean_euclid(X, sample_weight=sample_weight)
    #elif p == 0:
    elif p == 0 or (p < 0.01 and p > -0.01): #Anton1: added (p < 0.01 and p>-0.01) for when p=0.001 instead of 0
        return mean_riemann(X, 
                            sample_weight = sample_weight, 
                            init          = init,   #Anton2: added init
                            tol           = zeta,   #Anton3: added zeta here decreases the number significant digits
                            maxiter       = maxiter #increased from default 50 to 100
                            )
    elif p == -1:
        return mean_harmonic(X, sample_weight=sample_weight)

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
    else:
        warnings.warn("Power mean convergence not reached")

    M = K.conj().T @ K
    if p > 0:
        M = np.linalg.inv(M)

    return M
        