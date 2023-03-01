# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:07:48 2023

@author: antona

This is a dataset from Opne Food Facts
Link: https://drive.google.com/file/d/1NAtW9z2ymkRFia-P_FpkkIIsfnny0-Kp/view
"""

import pandas as pd
import os

#load data from a local 
path = os.path.join(os.getcwd(), "OFF_Francedataset.csv")
df = pd.read_csv(path)

#Display the first 10 rows
result = df.head(10)
print("First 10 rows of the DataFrame:")
print(result)

#select only a fraction of the data for analysis
#which column is the target ? : "nutriscore_grade" ?
#remove columns that contain the same data (and especially the ones that contain the target column)
#create a dataset only from the data where the target column is populated
#separate it in train and test 