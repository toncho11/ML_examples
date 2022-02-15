# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:26:58 2022

@author: antona
"""

#https://www.machinelearningplus.com/statistics/mahalanobis-distance/

import pandas as pd
import scipy as sp
import numpy as np

filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv'
df = pd.read_csv(filepath).iloc[:, [0,4,6]]
print(df.head())

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

#contains the vectors
df_x = df[['carat', 'depth', 'price']].head(1)

#data is the entire dataset 
df_x['mahala'] = mahalanobis(x=df_x, data=df[['carat', 'depth', 'price']])
print(df_x.head())

from scipy.stats import chi2
print("chi2: ", chi2.ppf((1-0.01), df=2))
#> 9.21

# Compute the P-Values
df_x['p_value'] = 1 - chi2.cdf(df_x['mahala'], 2)

print("Extreme values with a significance level of 0.01")
print(df_x.loc[df_x.p_value < 0.01].head(10))