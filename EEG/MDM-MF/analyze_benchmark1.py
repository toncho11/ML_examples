# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:17:29 2024

This script does some further analysis of Benchmark 1

@author: antona
"""

import pandas as pd
import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)

from matplotlib import pyplot as plt

datasets_P300 = ['BrainInvaders2013a', 
                 'BNCI2014-008', 
                  'BNCI2014-009', 
                  'BNCI2015-003', 
                  'BrainInvaders2015a', 
                  'BrainInvaders2015b', 
                  'Sosulski2019', 
                  'BrainInvaders2014a', 
                  'BrainInvaders2014b', 
                  'EPFLP300'
                 ]
datasets_P300_selected = ['BrainInvaders2013a_P', 
                 'BNCI2014-008_P', 
                 'BNCI2014-009_P', 
                 'BNCI2015-003_P', 
                 'BrainInvaders2015a_P', 
                 'BrainInvaders2015b_P', 
                 #'Sosulski2019_P', #problem with the statistics for Sosulski2019_P
                 'BrainInvaders2014a_P', 
                 'BrainInvaders2014b_P', 
                  # 'EPFLP300_P'
                 ]
datasets_MI = [ 'BNCI2015-004',  #5 classes, 
                'BNCI2015-001',  #2 classes
                'BNCI2014-002',  #2 classes
                'AlexMI',        #3 classes, Error: Classification metrics can't handle a mix of multiclass and continuous targets
              ]
datasets_LR = [ 'BNCI2014-001',
                'BNCI2014-004',
                'Cho2017',      #49 subjects
                'GrosseWentrup2009',
                'PhysionetMotorImagery',  #109 subjects
                'Shin2017A', 
                'Weibo2014', 
                'Zhou2016',
              ]


results = pd.read_csv('C:\\Users\\antona\\Desktop\\results\\MDM-MF-results\\WithinSessionBenchmark1_MDM_MDMMF_MDMMFLDA_SVM_LR_L1_L2_GPR_2015a_2015b\\results_dataframe.csv')  

removeP300  = False
removeMI_LR = True
def AdjustNames(df):
    
    for ind in df.index:
        if (df['dataset'][ind] in datasets_P300):
            df['dataset'][ind] = df['dataset'][ind] + "_P"
            
        if (df['dataset'][ind] in datasets_MI or df['dataset'][ind] in datasets_LR): 
            df['dataset'][ind] = df['dataset'][ind] + "_M"
    
    if (removeP300):
        df = df.drop(df[df['dataset'].str.endswith('_P', na=None)].index)
            
    if (removeMI_LR):
        df = df.drop(df[df['dataset'].str.endswith('_M', na=None)].index)
        
    if (removeP300 == False and removeMI_LR == True):
        for ind in df.index:
            if (df['dataset'][ind] not in datasets_P300_selected):
                df.drop(ind, inplace=True)
            
    return df

results = AdjustNames(results)


#generate statistics for the summary plot
#Compute matrices of p-values and effects for all algorithms over all datasets via combined p-values and
#combined effects methods
stats = compute_dataset_statistics(results)
P, T = find_significant_differences(stats)
#agg = stats.groupby(['dataset']).mean()
#print(agg)
#print(stats.to_string()) #not all datasets are in stats

#negative SMD value favors the first algorithm, postive SMD the second
#A meta-analysis style plot that shows the standardized effect with confidence intervals over
#all datasets for two algorithms. Hypothesis is that alg1 is larger than alg2
fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LDA", "MDM")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LDA", "MDM_MF")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM", "MDM_MF")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF", "MDM_MF_LR_l2")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF", "MDM_MF_LR_l1") #MDM_MF_LR_l1 best for P300
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LR_l1", "MDM_MF_LR_l2")
plt.show()

fig = moabb_plt.meta_analysis_plot(stats, "MDM_MF_LDA", "MDM_MF_LR_l1")
plt.show()

#summary plot - significance matrix to compare pipelines.
#Visualize significances as a heatmap with green/grey/red for significantly higher/significantly lower.
moabb_plt.summary_plot(P, T)
plt.show()

#MDM_LDA and MDM_MF_L2 are the best, MDM_MF_L2 better overall P300/MI, LDA excels more in MI and less in P300
#results.to_csv('C:\\Users\\antona\\Desktop\\results\\MDM-MF-results\\WithinSessionBenchmark1_MDM_MDMMF_MDMMFLDA_SVM_LR_L1_L2_GPR\\results_dataframe.csv', index=True)

#MDM_MF_LR_l1 best for P300
#MDM_MF_LDA - best for MI/LR
#MDM_MF_LDA - best for all cases
