# source https://towardsdatascience.com/visualising-the-classification-power-of-data-54f5273f640
# modified version by A. ANDREEV

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
import statsmodels.api as sm


cancer = load_breast_cancer()

df = pd.DataFrame(cancer['data'],columns=cancer['feature_names']) #len(cancer['feature_names']) = 30
df['y'] = cancer['target']
# 31 feaures/columns in 569 rows
print("Number of features: ", len(df.columns)) # y is also counted as feature!
print("Number of samples: ", df.shape[0])

#Scale the data (some features will tend to have higher variances because of their scale)
scaler = StandardScaler()
scaler.fit(df)
scaled = scaler.transform(df)

#Obtain principal components
pca = PCA().fit(scaled)

#Apply dimensionality reduction to scaled. scaled is projected on the first principal components previously extracted from a training set.
pc = pca.transform(scaled) 

#so in the new features (that are not the same as the original ones)
#usually the first 2-3 contain most of the variance
pc1 = pc[:,0]
pc2 = pc[:,1]


#Plot principal components
plt.figure(figsize=(10,10))

colour = ['#ff2121' if y == 1 else '#2176ff' for y in df['y']]
plt.scatter(pc1,pc2 ,c=colour,edgecolors='#000000')
plt.ylabel("Glucose",size=20)
plt.xlabel('Age',size=20)
plt.yticks(size=12)
plt.xticks(size=12)
plt.xlabel('PC1')
plt.ylabel('PC2')

# scree plot of the principal componenets
var = pca.explained_variance_[0:10] #percentage of variance explained
labels = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

plt.figure(figsize=(15,7))
plt.bar(labels,var,) #first param is horizontal
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')

#using groups, testing which group of features would be more useful for making predictions
group_1 = ['mean symmetry', 'symmetry error','worst symmetry',
'mean smoothness','smoothness error','worst smoothness']
        
group_2 = ['mean perimeter','perimeter error','worst perimeter', 
'mean concavity','concavity error','worst concavity']

#group 2 should have better classfication based on PCA scatter plot

groups = [group_1, group_2]

for i,g in enumerate(groups): #index, group

    x = df[g]
    x = sm.add_constant(x) #add a column (as first column) with ones 
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, 
                                                        random_state = 101)

    model = sm.Logit(y_train,x_train).fit() #fit logistic regression model

    predictions = np.around(model.predict(x_test)) 
    accuracy = metric.accuracy_score(y_test,predictions)
    
    print("Accuracy of Group {}: {}".format(i+1,accuracy))