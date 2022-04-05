#source https://machinelearningmastery.com/random-forest-ensemble-in-python/

# Random forest ensemble is an ensemble of decision trees and a natural extension of bagging.

# In bagging, a number of decision trees are created where each tree is created from a different bootstrap sample of the
# training dataset. A bootstrap sample is a sample of the training dataset where a sample may appear more than once in the
# sample, referred to as sampling with replacement.

# Bagging is an effective ensemble algorithm as each decision tree is fit on a slightly different training dataset, and in turn,
# has a slightly different performance. Unlike normal decision tree models, such as classification and regression trees (CART), 
# trees used in the ensemble are unpruned, making them slightly overfit to the training dataset. This is desirable as it helps to 
# make each tree more different and have less correlated predictions or prediction errors.

# Unlike bagging, random forest also involves selecting a subset of input features (columns or variables) at each split point in
# the construction of trees. Typically, constructing a decision tree involves evaluating the value for each input variable in 
# the data in order to select a split point. By reducing the features to a random subset that may be considered at each split 
# point, it forces each decision tree in the ensemble to be more different.

# The effect is that the predictions, and in turn, prediction errors, made by each tree in the ensemble are more different or 
# less correlated. When the predictions from these less correlated trees are averaged to make a prediction, it often results 
# in better performance than bagged decision trees.

# Another claim is that random forests “cannot overfit” the data. It is certainly true that increasing [the number of trees] 
# does not cause the random forest sequence to overfit.

# 1) Perhaps the most important hyperparameter to tune for the random forest is the number of random features to consider at each split point.
# A good heuristic for classification is to set this hyperparameter to the square root of the number of input features.
# It is set via the max_features argument

# 2) Another important hyperparameter to tune is the depth of the decision trees. Depths from 1 to 10 levels may be effective.
# It can be specified via the max_depth argument and is set to None (no maximum depth) by default.

# 3) The number of decision trees in the ensemble can be set. Often, this is increased until no further improvement is seen.
# It is set via the “n_estimators” argument and defaults to 100.

# 4) The “max_samples” argument can be set to a float between 0 and 1 to control the percentage of the size of the training 
# dataset to make the bootstrap sample used to train each decision tree.

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import sklearn

print(sklearn.__version__)

#=========================================================================
# Random Forest for Classification
#=========================================================================

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3, n_classes = 2)

# define the model
model = RandomForestClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))