# discretize numeric input variables
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)

# summarize data before the transform
print(X[:3, :])

# define the transform
trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')

# transform the data
X_discrete = trans.fit_transform(X)

# summarize data after the transform
print(X_discrete[:3, :])