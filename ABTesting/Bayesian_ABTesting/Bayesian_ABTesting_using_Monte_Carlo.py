# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:06:35 2023

Source: https://medium.com/vptech/introduction-to-bayesian-a-b-testing-in-python-df81a9b3f5fd

This is an example of Bayesian A/B testing (Bayesian Analysis) using the Monte Carlo sampling.

Here we calculate over a certain time period (70 days)
    - the probability to make the good choice with B
    - expected loss when choosing B over A
          - Maximum accepted loss is set to: 0.0001

Main advantages of the Bayesian method:
    - Bayesian A/B testing may always lead to usable results, whatever the volumes
    - Interesting results may be reached sooner than with the frequentist approach

"""

import pandas as pd
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_conversion_rate import data #data is provided in a .py file

# Monte Carlo integration (importance Sampling)
N_mc = 100000

proba_b_better_a = []
proba_b_better_a_error = []
expected_loss_a = []
expected_loss_a_error = []
expected_loss_b = []
expected_loss_b_error = []

for day in range(1, 70):

    mean_a, var_a = stats.beta.stats(a=1+sum(data['variation_a']['c'][:day]), 
                                     b=1+(sum(data['variation_a']['n'][:day])-sum(data['variation_a']['c'][:day])), 
                                     moments='mv')

    mean_b, var_b = stats.beta.stats(a=1+sum(data['variation_b']['c'][:day]), 
                                     b=1+(sum(data['variation_b']['n'][:day])-sum(data['variation_b']['c'][:day])), 
                                     moments='mv')

    randx_a = np.random.normal(loc=mean_a, 
                             scale=1.25*np.sqrt(var_a), 
                             size=N_mc)
    randx_b = np.random.normal(loc=mean_b, 
                             scale=1.25*np.sqrt(var_b), 
                             size=N_mc)

    f_a = stats.beta.pdf(randx_a,
                       a = 1+sum(data['variation_a']['c'][:day]), 
                       b = 1+(sum(data['variation_a']['n'][:day])-sum(data['variation_a']['c'][:day])))
    f_b = stats.beta.pdf(randx_b,
                       a = 1+sum(data['variation_b']['c'][:day]), 
                       b = 1+(sum(data['variation_b']['n'][:day])-sum(data['variation_b']['c'][:day])))

    g_a = stats.norm.pdf(randx_a,
                           loc=mean_a, 
                           scale=1.25*np.sqrt(var_a))
    g_b = stats.norm.pdf(randx_b,
                           loc=mean_b, 
                           scale=1.25*np.sqrt(var_b))

    y = (f_a * f_b) / (g_a * g_b)

    y_b = y[randx_b>=randx_a]

    p = 1/N_mc * sum(y_b)
    perr = np.sqrt(1*(y_b*y_b).sum()/N_mc - (1*y_b.sum()/N_mc)**2)/np.sqrt(N_mc)

    y_loss_a = ((randx_b-randx_a)*y)[randx_b>=randx_a]
    loss_A = 1/N_mc * sum(y_loss_a)
    loss_A_err = np.sqrt(1*(y_loss_a*y_loss_a).sum()/N_mc - (1*y_loss_a.sum()/N_mc)**2)/np.sqrt(N_mc)
    
    y_loss_b = ((randx_a-randx_b)*y)[randx_a>=randx_b]
    loss_B = 1/N_mc * sum(y_loss_b)
    loss_B_err = np.sqrt(1*(y_loss_b*y_loss_b).sum()/N_mc - (1*y_loss_b.sum()/N_mc)**2)/np.sqrt(N_mc)    
    
    proba_b_better_a.append(p)
    proba_b_better_a_error.append(perr)
    
    expected_loss_a.append(loss_A)
    expected_loss_a_error.append(loss_A_err)
    
    expected_loss_b.append(loss_B)
    expected_loss_b_error.append(loss_B_err)
    
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].set_ylim(0,1)
axs[0].plot(range(1, 70), proba_b_better_a)
axs[0].fill_between(range(1, 70), 
                 np.array(proba_b_better_a) - np.array(proba_b_better_a_error), 
                 np.array(proba_b_better_a) + np.array(proba_b_better_a_error), 
                 alpha = .5)

axs[1].plot(range(1, 70), expected_loss_a, label = 'Choosing A')
axs[1].fill_between(range(1, 70), 
                 np.array(expected_loss_a) - np.array(expected_loss_a_error), 
                 np.array(expected_loss_a) + np.array(expected_loss_a_error), 
                 alpha = .5)

axs[1].plot(range(1, 70), expected_loss_b, label = 'Choosing B')
axs[1].fill_between(range(1, 70), 
                 np.array(expected_loss_b) - np.array(expected_loss_b_error), 
                 np.array(expected_loss_b) + np.array(expected_loss_b_error), 
                 alpha = .5)

axs[1].hlines(.0001, 0, 70, color='black', label='Maximum accepted loss (0.0001)')

axs[0].set_ylabel('$P(\lambda_B>\lambda_A)$')
axs[0].set_xlabel('day')
axs[1].set_xlabel('day')
axs[1].set_ylabel('Expected loss')

axs[1].legend()

plt.show()

print("After only 30 days of A/B test, the probability to make the good choice with B had been around 80% for already 12 days, and the associated expected loss was far below the drop threshold of 0.0001. Variant B could have been chosen at this moment.")