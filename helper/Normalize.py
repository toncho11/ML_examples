from sklearn import preprocessing
import numpy as np

x_array = np.array([2,3,5,6,7,4,8,7,6]) # it can be a matrix (n_samples, n_features)

normalized_arr = preprocessing.normalize([x_array]) #needs another list []

print(normalized_arr[0])

#==============================================================

from sklearn import preprocessing
import pandas as pd
housing = pd.read_csv("https://dl.google.com/mlcc/mledu-datasets/california_housing_train.csv")

d = preprocessing.normalize(housing) #d is numpy.ndarray

scaled_df = pd.DataFrame(d) #convert back to frame
print(scaled_df.head())