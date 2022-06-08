# example of normalizing input data
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# define dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)

# summarize data before the transform

print(X[:3, :]) #first 3 rows

# define the scaler
trans = MinMaxScaler()

# transform the data
X_norm = trans.fit_transform(X)

# summarize data after the transform
print(X_norm[:3, :]) #first 3 rows

#MinMaxScaler is applied column-wise, Normalizer is applied row-wise.
#The Normalizer class from Sklearn normalizes samples individually to unit norm. It is not column based but a row-based normalization technique.
#So it your features are represented by columns (which is often the case) then you should use MinMaxScaler