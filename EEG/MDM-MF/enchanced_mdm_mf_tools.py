# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:08:36 2024

@author: antona
"""

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.spatialfilters import CSP
from pyriemann.utils.ajd import rjd, ajd_pham
import numpy as np

#it applies CSP only if the number of lectrodes is big e.x >=60
class CustomCspTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, metric_p, nfilter = 4):
        self.metric_p = metric_p
        self.nfilter = nfilter
    
    def fit(self, X, y=None):
        #print("fit csp")
        if X.shape[1] < 60:
            
            #n_iter_max=200 needs to be set in the code for "ale"
            self.csp = CSP(metric = "ale",    nfilter=self.nfilter, log=False)
        else:
            self.csp = CSP(metric = "euclid", nfilter=self.nfilter, log=False)
            
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
    
    def __init__(self, diag_method="rjd", norm_method="linalg"):
        self.diag_method = diag_method
        self.norm_method = norm_method
        
    def fit(self, X, y=None):
        
        #print("fit strict diagonalizer")
        
        if self.diag_method == "rjd":
            
            self.V = rjd(X,n_iter_max=200) #uses a modified version where D is not calculated to save time
        
        elif self.diag_method == "ajd_pham":
            
            V,_ = ajd_pham(X,)
            
            V = V.transpose()
            
            if self.norm_method == "L2":
                
                #L2 calculate the L2 norm of a vector, take the square root of the sum of the squared vector values
                #Normalize matrix in L2 norm on the columns
                rows, cols = len(V), len(V[0])
                for col in range(cols):
                    length = sum(V[row][col]**2 for row in range(rows)) ** 0.5
                    for row in range(rows):
                        V[row][col] /= length
                        
                self.V = V
                
                if 1 - (np.sqrt(np.sum(V * V))) > 0.0001:
                    raise Exception("Columns not converted to L2 norm!")
                
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
        