# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:19:49 2022

#Kolmogorov-Smirnov test
#non parametric test - does not require the assumption of narmality of the data
#can be used with one sample (a data vector) compared to a known distribution (with specific parameters)
#or to compare two samples (two data vectors) if they follow the same distribution

@author: antona
"""
import numpy as np
from scipy import stats

print("Example 1")
rng = np.random.default_rng()

#Here we show the "one-sample test"

#the data is generated using the uniform distribution
#and compared to a known distribution - the normal distribution

#H0 - both follow the same distribution
#H1 - they do not follow the same distribution

#test Kolmogorov smirnov
statistic, pvalue = stats.kstest(stats.uniform.rvs(size=100, random_state=rng), stats.norm.cdf)

if pvalue < 0.5:
    print("Correct")
    print("H0 is rejected, so this means that H1 is in effect which states that they do not follow the same distribution")
    print("This is correct because the data is generated from the Uniform distribution, but in the test we compare to the Normal distribution")
else:
    print("Not correct")
    
#if we replace the second paramter of the ktest with "stats.uniform.cdf" then H0 hypothesis will not be rejected.

#==========================================================================================================
print("Example 2")
import numpy as np

#Hontants des sinistres
M_sinis=np.array([3,9,23,19,3,4,18,14,10,10])

from scipy.stats import kstest

lbda=1/3

#testing the M_sinis vs an exponential distribution with specified lambda
#H0 - both follow the same distribution
#H1 - they do not follow the same distribution

#test Kolmogorov smirnov
stat_test,p_value=kstest(M_sinis,'expon',args=(0,lbda))

#print("la statistique de test est :",stat_test)
#print("la p_value du test est ", p_value)

if p_value < 0.5:
    print("H0 is rejected, so this means that H1 is in effect which states that they do not follow the same distribution")
else:
    print("H0 is not rejected - so they follow different distributions")