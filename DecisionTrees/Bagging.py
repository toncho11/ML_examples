#source https://machinelearningmastery.com/bagging-ensemble-with-python/

#Bagging is an ensemble machine learning algorithm that combines the predictions from many decision trees.
#Provides the basis for a whole field of ensemble of decision tree algorithms such as the popular Random Forest.

#Bagging ensemble is an ensemble created from decision trees fit on different samples of a dataset.

#Bootstrap Aggregation = Bagging

#An example of using bootstrap sampling would be estimating the population mean from a small dataset. 
#Multiple bootstrap samples are drawn from the dataset, the mean calculated on each, then the mean of the
# estimated means is reported as an estimate of the population.

#Predictions are made for regression problems by averaging the prediction across the decision trees. 
#Predictions are made for classification problems by taking the majority vote prediction for the classes 
#from across the predictions made by all the decision trees.

#The bagged decision trees are effective because each decision tree is fit on a slightly different training
#dataset, which in turn allows each tree to have minor differences and make slightly different skillful predictions.

#Bagging does not always offer an improvement. For low-variance models that already perform well, bagging can result in a decrease in model performance.

#The most famous extension of bagging is random forest.

#Bagging can be used with different algorithms, not just decision trees. One other example is knn specified via the “base_estimator” argument.
#In this case k value should be set to a low value or we can test different values of k to find the right balance of model variance to achieve good performance as a bagged ensemble.

##################################################################
# evaluate bagging algorithm for classification
##################################################################

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)

# define the model
model = BaggingClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

##################################################################
# evaluate bagging ensemble for regression
##################################################################

from sklearn.ensemble import BaggingRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=5)
# define the model
model = BaggingRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

##################################################################
# Bagging Hyperparameters
##################################################################

#Typically, the number of trees is increased until the model performance stabilizes. 
#Intuition might suggest that more trees will lead to overfitting, although this is not the case. 
#The number of trees can be set via the “n_estimators” argument and defaults to 100.

#The example below explores the effect of the number of trees with values between 10 to 5,000.


# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=5)
	return X, y
 
# get a list of models to evaluate
def get_models():
	models = dict()
	# define number of trees to consider
	n_trees = [10, 50, 100, 500, 500, 1000, 5000]
	for n in n_trees:
		models[str(n)] = BaggingClassifier(n_estimators=n)
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate the model and collect the results
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	# evaluate the model
	scores = evaluate_model(model, X, y)
	# store the results
	results.append(scores)
	names.append(name)
	# summarize the performance along the way
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
# We can see the general trend of no further improvement beyond about 100 trees.
pyplot.boxplot(results, labels=names, showmeans=True) # whisker plot
pyplot.show()


#The default is to create a bootstrap sample that has the same number of examples
#as the original dataset. Using a smaller dataset can increase the variance of the 
#resulting decision trees and could result in better overall performance.

#The sample size can be 100% (number of examples as the original dataset) because we are constructing 
#our dataset (for the currrent decion tree) with replacement - otherwise we will get the same dataset 
#each time we use 100% of the dataset.

#The number of samples used to fit each decision tree is set via the “max_samples” argument.