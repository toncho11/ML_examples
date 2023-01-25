# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 07:02:56 2023

@author: antona

source: https://towardsdatascience.com/bayesian-a-b-testing-with-python-the-easy-guide-d638f89e0b8a

This is an example of AB Testing without a Monte Carlo sampling because in the case
of conjugate priors an exact solution already exist (Exact Calculation of Beta Inequalities, John Cook, 2005)

"""

from scipy.stats import beta
import numpy as np
from calc_prob import calc_prob_between

#This is the known data: imporessions and conversions for the Control and Test set
imps_ctrl,convs_ctrl=16500, 30   #Control group
imps_test, convs_test=17000, 50  #Test group
#imps_test, convs_test=20500, 31  #Test group

#here we create the Beta functions for the two sets
#beta is the conjugate prior probability distribution for the Bernoulli
#beta has two parameters a and b
#The bera prior can be set as  Beta(s+1,(n−s)+1) where s is the number of successful attempts out of all (n)
#With Beta we get a PDF instead of a simple ratio (convs/imps)
a_C, b_C = convs_ctrl + 1, imps_ctrl - convs_ctrl + 1

beta_C = beta(a_C, b_C) #beta is a class from scipy stats

a_T, b_T = convs_test + 1, imps_test - convs_test + 1

beta_T = beta(a_T, b_T)

#end defining the two beta distributions

#calculating the lift
#‘uplift’ is how much the Test option increase (lifts) the CR with respect to the Control one
lift = (beta_T.mean() - beta_C.mean()) / beta_C.mean() #(Test - Control) / Control

#calculating the probability for Test to be better than Control
prob = calc_prob_between(beta_T, beta_C)

print (f"Test option lift Conversion Rates by {lift*100:2.2f}% with {prob*100:2.1f}% probability.")
#50% percent means that we are very uncertain
#output: Test option lift Conversion Rates by 59.68% with 98.2% probability