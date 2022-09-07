# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:02:52 2022

article: https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
source: https://gist.github.com/gsampath127/a44b56ece3ebb026e0570867e56a5b89

Perform Chi-Square test for Bank Churn prediction (find out different patterns 
on customer leaves the bank).

This example is for categorical predictor and categorical response.
For continous we need to use ANOVA feature selection. 

Dataset is from here: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
"""

import numpy as numpy
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

churn_df = pd.read_csv('bank.csv')

#Here we have 4 category predictors and one category response. 
#'Exited;, the response column, represents customer left the bank or not.
churn_df.head()

churn_df.drop("Surname", inplace=True, axis=1) #remove names as they are not useful

#encode string data to numeric 0,1,2,3, ...
label_encoder = LabelEncoder()
churn_df['Geography'] = label_encoder.fit_transform(churn_df['Geography'])
churn_df['Gender'] = label_encoder.fit_transform(churn_df['Gender'])

from sklearn.feature_selection import chi2

X = churn_df.drop('Exited',axis=1)
y = churn_df['Exited']

chi_scores = chi2(X,y) #here we use chi2 from sklearn, but tjere is also another one from scipy.stats
chi_scores

#here first array represents chi square values and second array represnts p-values
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()

#Since 'HasCrCard' has the highest p-value, it says that this variables is 
#independent of the repsone and can not be considered for model training