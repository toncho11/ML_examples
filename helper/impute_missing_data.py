# statistical imputation transform for the horse colic dataset
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')

# split into input and output elements
data = dataframe.values #converts to numpy.ndarray matrix 
ix = [i for i in range(data.shape[1]) if i != 23] #all indexes without 23
X, y = data[:, ix], data[:, 23] #row 23 is y, the rest is X

# print total missing
print('Missing: %d' % sum(isnan(X).flatten())) #A boolean object with the value of True evaluates to 1 in the sum() function

# define imputer
imputer = SimpleImputer(strategy='mean')

# fit on the dataset
imputer.fit(X)

# transform the dataset
X_new = imputer.transform(X)

# print total missing
print('Missing: %d' % sum(isnan(X_new).flatten()))