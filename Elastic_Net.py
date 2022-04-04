#source: https://machinelearningmastery.com/elastic-net-regression-in-python/

# A problem with linear regression is that estimated coefficients of the model can become large, 
# making the model sensitive to inputs and possibly unstable. 
# This is particularly true for problems with few samples (n) than input predictors (p) 
# or variables (so-called p >> n problems).

# We can change the loss function to include additional costs for a model that has large coefficients. 
# Linear regression models that use these modified loss functions during training are referred to collectively as penalized linear regression.

# Regularization works by biasing data towards particular values (such as small values near zero). 
# The bias is achieved by adding a tuning parameter to encourage those values:
    
# L1 penalty: it limits the size of the coefficients. L1 can yield sparse models (i.e. models with few coefficients); Some coefficients can become zero and eliminated.
# L2 penalty: it will not yield sparse models and all coefficients are shrunk by the same factor (none are eliminated).

# Elastic net is a penalized linear regression model that includes both the L1 and L2 penalties during training.

# We can use "alpha" to define elastic_net_penalty = (alpha * l1_penalty) + ((1 – alpha) * l2_penalty)
# For example, an alpha of 0.5 would provide a 50 percent contribution of each penalty to the loss function. An alpha value of 0 gives all weight to the L2 penalty and a value of 1 gives all weight to the L1 penalty.
# “lambda” parameter controls the weighting of the sum of both penalties to the loss function. A default value of 1.0 is used to use the fully weighted penalty; a value of 0 excludes the penalty. 
# elastic_net_loss = loss + (lambda * elastic_net_penalty)

# average mean absolute error (MAE)

# load and summarize the housing dataset

# evaluate an elastic net model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from numpy import arange
from sklearn.model_selection import GridSearchCV

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values

X, y = data[:, :-1], data[:, -1]

# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))

#=========================================================================
# Manual tuning hyper paramters: l1_ratio and alpha
#=========================================================================

# l1_ratio (scikitlearn) = "alpha" = 0.5 by default (in scikitlearn)
# alpha (scikitlearn)    = "lambda" = 1.0 by default (in scikitlearn)

# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = absolute(scores)
print('Mean MAE manual tuning: %.3f (%.3f)' % (mean(scores), std(scores)))

#=========================================================================
# Automatic Tuning of hyper paramters: l1_ratio and alpha
#=========================================================================

# define grid
grid = dict()
grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
grid['l1_ratio'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE automatic tuning: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

print("MAE decrease with with automatic tuning: ", mean(scores) - absolute(results.best_score_))