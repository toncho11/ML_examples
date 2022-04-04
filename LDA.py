# -*- coding: utf-8 -*-
"""
https://machinelearningmastery.com/linear-discriminant-analysis-with-python/

LDA description:
    
It is a linear classification algorithm, like logistic regression. 
This means that classes are separated in the feature space by lines or hyperplanes.

It works by calculating summary statistics for the input features by class label, 
such as the mean and standard deviation. These statistics represent the model learned 
from the training data. 

Predictions are made by estimating the probability that a new example belongs to each class 
label based on the values of each input feature. The class that results in the largest probability
is then assigned to the example. As such, LDA may be considered a simple application of Bayes Theorem for classification.

Data assumptions:

    - the observations within each class come from a normal distribution with a class-specific mean vector and a common variance
    - it also assumes that the input variables are not correlated
    
Preprocessing:
    - we recommend that predictors be centered and scaled and that near-zero variance predictors be removed
"""

#=========================================================================
# evaluate a lda model on the dataset on a synthetic dataset
#=========================================================================

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import arange
from sklearn.model_selection import GridSearchCV

# define a synthetic classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)

# define model
model = LinearDiscriminantAnalysis()

# define model evaluation method
# we can repeat the k-fold cross-validation process multiple times and report the mean performance across all folds and all repeats
# Stratified will balance the samples in each fold to an even number
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#=========================================================================
# make a prediction with the LDA model on the dataset
#=========================================================================

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)

# define model
model = LinearDiscriminantAnalysis()

# fit model
model.fit(X, y)

# define new data
row = [0.12777556,-3.64400522,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,1.01001752,0.56768485]

# make a prediction
yhat = model.predict([row])

# summarize prediction
print('Predicted Class: %d' % yhat)

#=========================================================================
# Tune hyper paramters: use different solvers ['svd', 'lsqr', 'eigen']
#=========================================================================

# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)


#=========================================================================
# Tune hyper paramters: use shrinkage
#=========================================================================

#Shrinkage adds a penalty to the model that acts as a type of regularizer, reducing the complexity of the model.
#Regularization reduces the variance associated with the sample based estimate at the expense of potentially increased bias.

# define model
model = LinearDiscriminantAnalysis(solver='lsqr') #specific solvers are required to use shrinkage
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid
grid = dict()
grid['shrinkage'] = arange(0, 1, 0.01)

# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X, y)

# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
