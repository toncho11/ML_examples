'''

Light GBM (Light Gradient Boosting Machine)

There has been only a slight increase in accuracy and auc score by applying
Light GBM over XGBOOST but there is a significant difference in the execution 
time for the training procedure. Light GBM is almost 7 times faster than 
XGBOOST and is a much better approach when dealing with large datasets. 

source: https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/

pip install lightgbm
'''

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Loading the Data : 
data = pd.read_csv('SVMtrain.csv')
#data.head()

#Loading the variables:
# To define the input and output feature
x = data.drop(['Embarked','PassengerId'],axis=1)
y = data.Embarked #.to_numpy(dtype='int')

#convert "sex" to mumeric
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
x["Sex"] = ord_enc.fit_transform(x[["Sex"]]).astype(int)

# train and test split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          eval_metric='logloss')

print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))