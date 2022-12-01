# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:36:27 2022

source: https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

To use Keras models with scikit-learn, you must use the KerasClassifier wrapper from the SciKeras module:
pip install scikeras[tensorflow]

"""

from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import numpy as np

# load dataset
dataframe = read_csv("sonar.csv", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_shape=(60,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# evaluate model with standardized dataset

estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=1)

kfold = StratifiedKFold(n_splits=5, shuffle=True)

results = cross_val_score(estimator, np.array(X), np.array(encoded_Y), cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))