# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:01:34 2023

Random Forest with NaNs. NaNs are imputed with sklearn's SimpleImputer.
"""

from __future__ import print_function

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


X_train = [[0, 0, np.nan], [np.nan, 1, 1]]
Y_train = [0, 1]
X_test_1 = [0, 0, np.nan]
X_test_2 = [0, np.nan, np.nan]
X_test_3 = [np.nan, 1, 1]

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

# Impute our data, then train
X_train_imp = imp.transform(X_train)
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train_imp, Y_train)

for X_test in [X_test_1, X_test_2, X_test_3]:
    # Impute each test item, then predict
    X_test_imp = imp.transform(X_test)
    print(X_test, '->', clf.predict(X_test_imp))
