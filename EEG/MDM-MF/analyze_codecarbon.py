# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:34:06 2024

You first need to install the package:
pip install codecarbon
And then run the evaluation

@author: antona
"""
import pandas as pd
from moabb import benchmark, set_log_level
from moabb.analysis.plotting import codecarbon_plot

results = pd.read_csv("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\results_dataframe_carbon.csv")
#results = pd.read_csv("C:\\Work\\PythonCode\\ML_examples\\EEG\\MDM-MF\\benchmark\\LeftRightImagery\\analysis\\data.csv")
order_list = results['pipeline'].unique().tolist()
print(order_list)

codecarbon_plot(results, order_list, country="(France)")

set_log_level("info")