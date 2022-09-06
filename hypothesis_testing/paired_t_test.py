# -*- coding: utf-8 -*-
"""

Source: https://www.marsja.se/how-to-use-python-to-perform-a-paired-sample-t-test/
    
The paired sample t-test is also known as the dependent sample t-test, and 
paired t-test. Furthermore, this type of t-test compares two averages (means)
and will give you information if the difference between these two averages
are zero. In a paired sample t-test, each participant is measured twice, 
which results in pairs of observations.

It is used for conducting a pre- and post-test study.

Hypotheses:
    
Now, when performing dependent sample t-tests you typically have the following two hypotheses:

- Null hypotheses: the true mean difference is equal to zero (between the observations), meaning they have identical average (expected) values
- Alternative hypotheses: the true mean difference is not equal to zero (two-tailed)

Note, in some cases we also may have a specific idea, based on theory, about
the direction of the measured effect. For example, we may strongly believe
(due to previous research and/or theory) that a specific intervention should
have a positive effect. In such a case, the alternative hypothesis will be
something like: the true mean difference is greater than zero (one-tailed). 

Assumptions
Before we continue and import data we will briefly have a look at the assumptions of this paired t-test. 
Now, besides that the dependent variable is on interval/ratio scale, and is 
continuous, there are three assumptions that need to be met.

- Are the two samples independent?
- Does the data, i.e., the differences for the matched-pairs, follow a normal distribution?
- Are the participants randomly selected from the population?

If your data is not following a normal distribution you can  transform your 
dependent variable using square root, log, or Box-Cox in Python.

"""

#pip install scipy pandas seaborn pingouin

print ("Version 1 with scipy.stats")

import pandas as pd

df = pd.read_csv('./paired_samples_data.csv', index_col=0)

from scipy.stats import ttest_rel

a = df.query('test == "Pre"')['score']
b = df.query('test == "Post"')['score']

statistic, pvalue = ttest_rel(a, b, alternative = "two-sided")

if pvalue < 0.05:
    print("Reject null hypothesis => There is a change in the two groups")
else:
    print("Failed to reject null hypothesis")

statistic, pvalue = ttest_rel(a, b, alternative = "greater") #post is higer than pre

if pvalue < 0.05:
    print("Reject null hypothesis => The pre condition is higher than the post condition")
else:
    print("Failed to reject null hypothesis")


statistic, pvalue = ttest_rel(a, b, alternative = "less") #post is higer than pre

if pvalue < 0.05:
    print("Reject null hypothesis => The pre condition is lower than the post condition")
else:
    print("Failed to reject null hypothesis")

print("Mean of pre condition",  a.mean())
print("Mean of post condition", b.mean())

print ("Version 2 with pingouin")

#pingouin provides more statistics about the data compared to scipy.stats.ttest_rel
import pingouin as pt

# Python paired sample t-test:
statistics = pt.ttest(a, b, paired=True, alternative="less")

#statistics is a DataFrame
statistics.columns

pvalue = float(statistics["p-val"][0])
cohend = float(statistics["cohen-d"][0])
bayes_factor = float(statistics["BF10"][0])

if (pvalue < 0.001):
    print ("Values definitely increased in the second sample. Result is statistically significant")

#Cohen's d is defined as the difference between two means divided by a standard deviation for the data
if (cohend <= 0.2):
    print("Difference is trivial, even if it is statistically significant.")
else:
    print("Difference is 'medium' or 'large'")
    
if (bayes_factor > 100):
    print ("Very strong evidence for hypothesis H1 (currently that pre is lower than post)")
    #print("Bayes Factor {:.8f}".format(bayes_factor))
    
import seaborn as sns

#it automatically extract the two conditions "pre" and "post"
sns.boxplot(x='test', y='score', data=df)