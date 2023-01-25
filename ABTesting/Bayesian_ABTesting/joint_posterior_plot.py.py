# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:08:39 2023

Source: https://medium.com/vptech/introduction-to-bayesian-a-b-testing-in-python-df81a9b3f5fd

Shows a joint posterior plot.
This file is a part of the Bayesian A/B exmaple in Python

"""

import pandas as pd
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_conversion_rate import data

N_plot = 1000
day = 8

X = np.linspace(0.075,.08,N_plot)

posterior_a = stats.beta.pdf(X, 
                             1+sum(data['variation_a']['c'][:day]),
                             1+(sum(data['variation_a']['n'][:day])-sum(data['variation_a']['c'][:day])))
posterior_b = stats.beta.pdf(X, 
                             1+sum(data['variation_b']['c'][:day]),
                             1+(sum(data['variation_b']['n'][:day])-sum(data['variation_b']['c'][:day])))

A, B = np.meshgrid(posterior_a, posterior_b)

Z = A * B

f = plt.figure(figsize=(12,10))
gs0 = gridspec.GridSpec(5, 6, figure=f, wspace=.02, hspace=.02)
ax1 = f.add_subplot(gs0[1:, :-2])
ax2 = f.add_subplot(gs0[:1, :-2])
ax3 = f.add_subplot(gs0[1:, -2:-1])

ax1.contourf(X, X, Z, cmap= 'Blues')
ax1.plot(X, X, color = 'orange', label = '$\lambda_A = \lambda_B$')
ax2.plot(X, posterior_a)
ax3.plot(posterior_b, X)

ax1.set_xlabel('$\lambda_A$')
ax1.set_ylabel('$\lambda_B$')

ax3.set_axis_off()
ax2.set_axis_off()

ax1.legend()
plt.show()

print("in the figure variation B seems to be more promising than variation A because the main of the contour plot is above the orange line.")
