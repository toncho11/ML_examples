# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:22:49 2022

source1 https://www.statology.org/how-to-find-the-t-critical-value-in-python/
source2 https://github.com/eceisik/eip/blob/main/hypothesis_testing_examples.ipynb

Whenever you conduct a t-test, you will get a test statistic as a result. 
To determine if the results of the t-test are statistically significant, you 
can compare the test statistic to a T critical value. If the absolute value
of the test statistic is greater than the T critical value, then the results
of the test are statistically significant.
"""

import scipy.stats

import scikit_posthocs as sp
import numpy as np
from scipy import stats
import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format

def check_normality(data):
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality <0.05:
        print("Reject null hypothesis >> The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis >> The data is normally distributed")     

def check_variance_homogeneity(group1, group2):
    test_stat_var, p_value_var= stats.levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var <0.05:
        print("Reject null hypothesis >> The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis >> The variances of the samples are same.")

sync = np.array([94. , 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2,
       87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6])
asyncr =np.array([77.1, 71.7, 91. , 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2])

check_normality(sync)
check_normality(asyncr)

check_variance_homogeneity(sync, asyncr)

alpha = 0.05

#H₀: Ms <= Ma (we want to reject this)
#H₁: Ms >  Ma
#Thi is a right tailed test because Ms > Ma

#Is the sync group better than the async group
ttest,p_value = stats.ttest_ind(sync,asyncr,alternative='greater') #'greater' because H1: Ms > Ma and Ms is the first parameter
print("p value:%.8f" % p_value)

print("")
#Method 1====================================================================

if p_value < alpha:
    print("Rejected null hypothesis")
else:
    print("Failed to reject null hypothesis") 
    
#left tailed q = alpha
#right tailed q = 1-lapha
#two sided q=1-.05/2

#get the critical value of a right tailed test
critical_stat = scipy.stats.t.ppf(q=1-alpha,df=(len(sync)-1))

#Method 2====================================================================

if critical_stat < ttest:
    print("Rejected null hypothesis")
else:
    print("Failed to reject null hypothesis") 



