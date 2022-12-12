# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:46:12 2022

@author: antona
"""

from tpot import TPOTClassifier
import numpy as np

def LoadTrainTest():
    filename = 'C:\\Work\\PythonCode\\ML_examples\\EEG\\DataAugmentation\\UsingTimeVAE\\TrainTest.npz'
    print("Loading data from: ", filename)
    data = np.load(filename)
    
    return data['X_train'] , data['X_test'], data['y_train'], data['y_test']

if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test = LoadTrainTest()
    print(X_train.shape)
        
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2)
    
    pipeline_optimizer.fit(X_train, y_train)
    
    print(pipeline_optimizer.score(X_train, y_train))
    print(pipeline_optimizer.score(X_test, y_test))
    
    pipeline_optimizer.export('tpot_exported_pipeline.py')