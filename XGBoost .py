# -*- coding: utf-8 -*-
"""

An example of Boosting using the XGBoost Python library

source: https://www.datacamp.com/community/tutorials/xgboost-in-python

Here the XGBoost is used solve a regression problem (predict price). 

install: pip install xgboost
"""

from sklearn.datasets import load_boston
boston = load_boston()

import pandas as pd

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

data['PRICE'] = boston.target

import xgboost as xgb

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X, y = data.iloc[:,:-1],data.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=X,label=y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


