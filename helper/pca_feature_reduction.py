# example of pca for dimensionality reduction
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=7, random_state=1)

# summarize data before the transform
print(X[:3, :])

# define the transform
trans = PCA(n_components=3)

# transform the data
X_dim = trans.fit_transform(X)

# summarize data after the transform
print("With redeuced to only 3 features:")
print(X_dim[:3, :])