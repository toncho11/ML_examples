'''

Authors: Gregoire CATTAN, Anton ANDREEV

It is a POC code that changes the X (which is kind of normal) and y (usually immutable) when using ScikitLearn.
This allows for a DataAugmentation ScikitLearn transformer.

'''

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from random import sample

import numpy as np

class Handler():
    def __init__(self, array) -> None:
        self.array = np.array(array)
        if(not self.supports_np_asarray()):
            self.inject_handler_asarray()
    
    def inject_handler_asarray(self):
        self.asarray0 = np.asarray
        np.asarray = lambda y, **params: y.array if type(y) is Handler else self.asarray0(y, **params)

    def restore_np_as_array(self):
        if hasattr(self, 'asarray0'):
            np.asarray = self.asarray0

    def supports_np_asarray(self):
        test = np.asarray(self)
        shape = np.shape(test)
        return not len(shape) == 0

    @property
    def shape(self):
        return self.array.shape

    def __str__(self) -> str:
        return str(self.array)
    
    def __getitem__(self, a):
        return self.array.__getitem__(a)

    def __setitem__(self, a, b):
        return self.array.__setitem__(a, b)

class DataAugmentation(TransformerMixin):
    def fit(self, X, y, params):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        #add two  extra sample
        sample1 = sample(range(10, 30), X.shape[1])
        sample2 = sample(range(10, 30), X.shape[1])
        X = np.append(X, [sample1, sample2], axis=0)
        
        y.array = np.append(y.array, 1) #label sample 1
        y.array = np.append(y.array, 0) #label sample 2
        
        return X

class Debug(TransformerMixin):
    def fit(self, X, y, **params):
        print("Debug_fit:",X.shape, y.shape)
        return self

    def transform(self, X):
        print("Debug_transform:",X.shape, y.shape)
        return X

#X = np.array([[100], [100]])
#y = np.array([0, 0])
X, y = make_classification(n_samples=200, random_state=0)
X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=0)

handler = Handler(y_train)

clf = make_pipeline(DataAugmentation(), Debug(), SVC())

clf.fit(X_train, handler)

#X_test = np.array([[100], [101]])
y_pred = clf.predict(X_test)

print("pred",y_pred)
print(y_pred.size)

handler.restore_np_as_array()
