# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:48:02 2023

@author: antona
"""

import pandas as pd
import os
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split

dataset_name = "OFF_Francedataset.csv"
#load data from a local 
path = os.path.join(os.getcwd(), dataset_name)
#df = pd.read_csv(path)

#read a subset of the data (only for faster experimentation)
df = pd.read_csv(path, sep=',', skiprows=0, nrows=50000)

#Display the first 10 rows
result = df.head(10)
print("First 10 rows of the DataFrame:")
print(result)

#select only a fraction of the data for analysis
#subset = df.sample(n = 5000)

#which column is the target 
#the target is the nuneric column "nutriscore_score" (numeric) which allows for the calculation of the final "nutriscore_grade"
print(df.columns.tolist())

search_columns = [col for col in df.columns.tolist() if 'nutriscore_score' in col]
#print(search_columns)

#nutrition-score-fr_100g, nutrition-score-uk_100g, nutriscore_score

#df['nutrition-score-uk_100g'].isnull().count()
#nutrition-score-uk_100g is ann NaN

target_column = "nutriscore_score" #nutriscore_score or nutriscore_grade

#check if two columns are the same
print(df["nutrition-score-fr_100g"].equals(df["nutriscore_score"]))
#so we can use nutriscore_score

#remove columns that contain the same data (and especially the ones that contain the target column)

columns_100g = [col for col in df.columns.tolist() if '_100g' in col]
print(len(columns_100g))
columns_100g = [col for col in columns_100g if not 'nutri' in col]
print(len(columns_100g))

list_new_columns = columns_100g + [target_column]
df_new = df[list_new_columns]

print(list_new_columns)

#select only the rows with valid nutri score
df_new = df_new[~df_new[target_column].isna()]

#======================================================================================

y = df_new[target_column]
X = df_new.loc[:, df_new.columns != target_column]

#X_train, X_test, y_train, y_test = train_test_split(X.index,y,test_size=0.2)
#X.iloc[X_train] # return dataframe train

# define model
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators = 1000) #larger n_estimators increases performance 

# fit model
model.fit(X, y)

# define test data
n = 56
row = X.iloc[[n]]
new_data = row.to_numpy()

# make a prediction
yhat = model.predict(new_data)

# summarize prediction
print('Predicted vs true value:', yhat, " vs ", y.iloc[[n]].to_numpy())