# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:07:48 2023

@author: antona

This is a dataset from Open Food Facts
Link: https://drive.google.com/file/d/1NAtW9z2ymkRFia-P_FpkkIIsfnny0-Kp/view
Fields: https://static.openfoodfacts.org/data/data-fields.txt
"""

import pandas as pd
import os

dataset_name = "OFF_Francedataset.csv"
#load data from a local 
path = os.path.join(os.getcwd(), dataset_name)
#df = pd.read_csv(path)

#read a subset of the data (only for faster experimentation)
df = pd.read_csv(path, sep=',', skiprows=0, nrows=5000)

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
print(search_columns)

#nutrition-score-fr_100g, nutrition-score-uk_100g, nutriscore_score

#df['nutrition-score-uk_100g'].isnull().count()
#nutrition-score-uk_100g is ann NaN

#check if two columns are the same
print(df["nutrition-score-fr_100g"].equals(df["nutriscore_score"]))
#so we can use nutriscore_score

#remove columns that contain the same data (and especially the ones that contain the target column)

columns_100g = [col for col in df.columns.tolist() if '_100g' in col]
print(len(columns_100g))
columns_100g = [col for col in columns_100g if not 'nutri' in col]
print(len(columns_100g))

list_new_columns = columns_100g# + ['nutriscore_score']
df_new = df[list_new_columns]

#create a dataset only from the data where the target column is populated
#find which columns are actually most useful

#df_new_0 = df_new.fillna(0) #puy 0 for NaNs - not logical

#separate it in train and test 

#Classification

#https://stackoverflow.com/questions/30317119/classifiers-in-scikit-learn-that-handle-nan-null
#apply XGBoost

#another option is similarity search - find the one closest in parameters that has a nutriscore_score

