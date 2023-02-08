# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:21:57 2023

@author: antona

AB Testing
First we need a specification. What difference is considered meanigful?
First we calculate "effect size" and then "samples needed". 

We set the power parameter to 0.8 and in practice this means that if there exists an actual 
difference in conversion rate between our designs, assuming the difference is the one we 
estimated (13% vs. 15%), we have about 80% chance to detect it as statistically significant 
in our test with the sample size we calculated ("samples needed").

source: https://towardsdatascience.com/ab-testing-with-python-e5964dd66143

"""

# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

# Some plot styling preferences
plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica',
        'weight' : 'bold',
        'size'   : 14}

mpl.rc('font', **font)

#1) Calculate based on our requirements how much samples do we need

effect_size = sms.proportion_effectsize(0.13, 0.15) # Calculating effect size based on our expected rates

# Calculating sample size
required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.8, # The probability of finding a statistical difference between the groups in our test when a difference is actually present
    alpha=0.05, #the p value calculated later should be lower than alpha in order to 
    ratio=1
    ) 

required_n = ceil(required_n) # Rounding up to next whole number                          

print(required_n)

#2) Load the actual data and cleaning
df = pd.read_csv('ab_data.csv') #check for zip in folder

df.head()

#Cleaning data
session_counts = df['user_id'].value_counts(ascending=False)
#multi_users = session_counts[session_counts > 1].count()

#print(f'There are {multi_users} users that appear multiple times in the dataset')

users_to_drop = session_counts[session_counts > 1].index

df = df[~df['user_id'].isin(users_to_drop)]
print(f'The updated dataset now has {df.shape[0]} entries')

#3) Perform sampling based on the previously calcualted required_n

control_sample = df[df['group'] == 'control'].sample(n=required_n, random_state=22)
treatment_sample = df[df['group'] == 'treatment'].sample(n=required_n, random_state=22)

ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)

print(ab_test['group'].value_counts())

#4 Visualising the results

conversion_rates = ab_test.groupby('group')['converted']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']

conversion_rates.style.format('{:.3f}')
plt.figure(figsize=(8,6))
sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)

plt.ylim(0, 0.17)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15);

#4) Testing the hypothesis

from statsmodels.stats.proportion import proportions_ztest, proportion_confint
control_results = ab_test[ab_test['group'] == 'control']['converted']
treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']

n_con = control_results.count()
n_treat = treatment_results.count()

successes = [control_results.sum(), treatment_results.sum()] #the number of conversions (out of the total number of observations)
nobs = [n_con, n_treat] #total Number of Observations

z_stat, pval = proportions_ztest(successes, nobs=nobs)

#calucalte confident intervals
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')

if pval < 0.5:
    print("H0 rejected.") #We have enough statistical evidence to support the alternative claim (H1)
else:
    print("H0 not rejected.")
    print("This means that our new design did not perform significantly different (let alone better) than our old one.")
    