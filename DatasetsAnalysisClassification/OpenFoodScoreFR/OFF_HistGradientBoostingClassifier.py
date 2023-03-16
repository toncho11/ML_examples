# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:07:02 2023

@author: antona

The objective is to learn to calculate the nutriscore_score column.
We select the columns with "100g" as X and column "nutriscore_score" as y.

Dataset is the French version of the Open Food Facts (OFF) dataset.

HistGradientBoostingClassifier is used for two reasons:
    - it can handle NaN values
    - it is fast
    
We accept that there are 65 classes between (-15 and 40). 
The classes are actaully reduced to 50 for example because there are very few products
with certain values of the nutri score. For example nutri scores [-15, -14, 37, 40] are
rearly encountered.

With default parameters the accuracy is low.

"""

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

dataset_name = "OFF_Francedataset.csv"
#load data from a local 
path = os.path.join(os.getcwd(), dataset_name)

#read a subset of the data (only for faster experimentation)
n_rows_read = 500000 * 5 #more data increases the classification score

df = pd.read_csv(path, sep=',', skiprows=0, nrows=n_rows_read)

#some checks
if df["nutriscore_score"].min() < -15 or df["nutriscore_score"].max() > 40:
    print("Error range nutri score")
    import sys
    sys.exit()

#select the data 
target_column = "nutriscore_score" #nutriscore_score or nutriscore_grade

#select 100g columns
columns_100g = [col for col in df.columns.tolist() if '_100g' in col]

#remove 'nutrition-score-fr_100g', 'nutrition-score-uk_100g'
columns_100g = [col for col in columns_100g if not 'nutri' in col]
print("Number of columns to be used for X:", len(columns_100g))

list_new_columns = columns_100g + [target_column]
df_new = df[list_new_columns]

print(list_new_columns)

#select only the rows with valid nutri score to be used for train and validation
df_new = df_new.dropna(how = "any", subset=[target_column])

#make the 15 + 40 = 55 classes
df_new[target_column] = df_new[target_column].astype('int')

#remove classes with very few examples (for both training and testing)
df_new_count_per_group = df_new.groupby([target_column])[target_column].count()
nutri_score_to_remove = df_new_count_per_group[df_new_count_per_group < 10].index.to_list()
df_new = df_new[df_new[target_column].isin(nutri_score_to_remove) == False]

print("Number of classes:", len(df_new[target_column].unique()), "/ 55")
#======================================================================================
mean_accuracy = []

df_new.index = df_new.index * 10

splits = 5

kf = KFold(n_splits = splits, shuffle = True)

#kfold split
for x in range(splits):
    
    result = next(kf.split(df_new), None)

    train = df_new.iloc[result[0]]
    test =  df_new.iloc[result[1]]

    train_X = train.loc[:, train.columns != target_column]
    train_y = train[target_column]
    
    test_X = test.loc[:, test.columns != target_column]
    test_y = test[target_column]
    
    # define model
    model = HistGradientBoostingClassifier()
    
    # fit model
    print("Training ...")
    model.fit(train_X, train_y)
    
    # make a prediction
    print("Evaluation ...")
    yhat = model.predict(test_X)

    acc = accuracy_score(test_y, yhat)
    print("Current accuracy:", x, acc)
    mean_accuracy.append(acc)
    
print("Mean accuracy: ", np.mean(mean_accuracy), "on", n_rows_read, "records (when considred as a classification task)")