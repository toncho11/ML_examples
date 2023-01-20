# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:39:20 2023

@author: antona

This is an example of handling imbalanced dataset.
It is using the BalancedBaggingClassifier
which is provided by the package "imbalanced-learn". 

Versions: 
    - imbalanced-learn 0.10.1
    - scikitlearn 1.2.0
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_validate
from imblearn.datasets import make_imbalance

df, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True, parser='auto')
df = df.drop(columns=["fnlwgt", "education-num"])
classes_count = y.value_counts()
scoring = ["accuracy", "balanced_accuracy"]
  
ratio = 30
df_res, y_res = make_imbalance(
    df,
    y,
    sampling_strategy={classes_count.idxmin(): classes_count.max() // ratio},
)

index = []
scores = {"Accuracy": [], "Balanced accuracy": []}

num_pipe = SimpleImputer(strategy="mean", add_indicator=True)
cat_pipe = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="missing"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
)

preprocessor_tree = make_column_transformer(
    (num_pipe, selector(dtype_include="number")),
    (cat_pipe, selector(dtype_include="category")),
    n_jobs=2,
)

bag_clf = make_pipeline(
    preprocessor_tree,
    BalancedBaggingClassifier(
        estimator=HistGradientBoostingClassifier(random_state=42),
        n_estimators=10,
        random_state=42,
        n_jobs=2,
    ),
)

index += ["Balanced bag of histogram gradient boosting"]
cv_result = cross_validate(bag_clf, df_res, y_res, scoring=scoring)
scores["Accuracy"].append(cv_result["test_accuracy"].mean())
scores["Balanced accuracy"].append(cv_result["test_balanced_accuracy"].mean())

df_scores = pd.DataFrame(scores, index=index)
print(df_scores)