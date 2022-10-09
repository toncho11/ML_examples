'''

GradientBoostingClassifier

Boosting
Boosting is a special type of Ensemble Learning technique that works by 
combining several weak learners(predictors with poor accuracy) into a strong
learner(a model with strong accuracy). This works by each model paying 
attention to its predecessor’s mistakes.

The two most popular boosting methods are:

- Adaptive Boosting (also called AdaBoost)
- Gradient Boosting (shown here)

source 1: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
source 2: https://towardsdatascience.com/gradient-boosting-classification-explained-through-python-60cc980eeb3d

Boosting:


Comparison:
https://analyticsindiamag.com/adaboost-vs-gradient-boosting-a-comparison-of-leading-boosting-algorithms/#:~:text=outliers%20than%20AdaBoost.-,Flexibility,Boosting%20more%20flexible%20than%20AdaBoost.
https://www.educba.com/random-forest-vs-gradient-boosting/
'''

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

df = pd.DataFrame(load_breast_cancer()['data'],
                  columns=load_breast_cancer()['feature_names'])
df['y'] = load_breast_cancer()['target']
df.head(5)

X,y = df.drop('y', axis=1),df.y

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index,val_index in kf.split(X):
    X_train,X_val = X.iloc[train_index],X.iloc[val_index],
    y_train,y_val = y.iloc[train_index],y.iloc[val_index],
    
gradient_booster = GradientBoostingClassifier(learning_rate = 0.1)
gradient_booster.get_params()

gradient_booster.fit(X_train, y_train)
print(classification_report(y_val, gradient_booster.predict(X_val)))

#f1 score ...