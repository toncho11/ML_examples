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
    def __init__(self, euclid = False, nfilter = 10): #maxiter_128 = 2, n_iter_max_128=2):
        self.euclid = euclid
        self.nfilter = nfilter
        #self.maxiter_128 = maxiter_128
        #self.n_iter_max_128 = n_iter_max_128
    
    def fit(self, X, y=None):
        #print("fit csp")
        if X.shape[1] <= 64: #default < 60
            
            #n_iter_max=200 needs to be set in the code for "ale"
            #good maxiter = 20, n_iter_max = 10
            #5 5 ok
            self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 10, n_iter_max = 8) # maxiter = 50, n_iter_max = 100)
        else:
            if self.euclid:
                self.csp = CSP(metric = "euclid", nfilter=self.nfilter, log=False) #pca par example
            else:
                self.csp = CSP(metric = "ale", nfilter=self.nfilter, log=False, maxiter = 2 , n_iter_max = 2) #pca par example
            
        self.csp.fit(X,y)
        
        return self
    
    def transform(self, X):
        X_transformed = self.csp.transform(X)
        return X_transformed
        
        # if X.shape[1] < 60:
        #     X_transformed = self.csp.transform(X)
        #     return X_transformed
        # else:
        #     return X
        
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
        