# -*- coding: utf-8 -*-
"""

Time series prediction model LMS algorithm

Anaconda 3 2021_05
Python 3.8.8
Spyder 4.2.5

Tutorial https://towardsdatascience.com/machine-learning-and-signal-processing-103281d27c4b

@author: antona
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Generate data
num_train_data = 4000
num_test_data = 1000
timestep = 0.1
tm =  np.arange(0, (num_train_data+num_test_data)*timestep, timestep);
y = np.sin(tm) + np.sin(tm*np.pi/2) + np.sin(tm*(-3*np.pi/2)) 
SNR = 10
ypn = y + np.random.normal(0,10**(-SNR/20),len(y))

plt.plot(tm[0:100],y[0:100])
plt.plot(tm[0:100],ypn[0:100],'r')
plt.show()
print("Red is the signal with noise.")

#LMS
M = 1000
L = 64
yrlms = np.zeros(M+L)
wn = np.zeros(L)
print(wn.shape, yrlms.shape)
mu = 0.005
for k in range(L,M+L):
  yrlms[k] = np.dot(ypn[k-L:k],wn)
  e = ypn[k]- yrlms[k]
  wn=wn+(mu*ypn[k-L:k]*e)

plt.plot(yrlms[600:800],'g')
plt.plot(y[600:800],'r')
plt.show()

print("Done. Red is the real signal. Green is the predicted.")