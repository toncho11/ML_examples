# The minimization problem for decision trees is known to be NP-hard.

# Decion tree algorithms:
    # ID3
        #most widely used algorithm 
        #uses information gain
            #InformationGain(feature) = Entropy(Dataset) - Entropy(feature)
        #can have more than 2 children
        #slightly more balanced trees than CART 
        #id3 implementation: https://github.com/svaante/decision-tree-id3
    # C4.5 s the successor to ID3
        #C5.0 is Quinlan’s latest version release under a proprietary license. It uses less memory and builds smaller rulesets than C4.5 while being more accurate.
    # CART (Classification and Regression Trees)
        #outputs a decision tree where each fork is a split in a predictor variable and each end node contains a prediction for the outcome variable.
        #uses RSS (Sum of squared residuals) + Gini index
        #favors larger partitions
        #only binary trees as children
        #greedy - often the most frequent class is in its own branch
        #optimized version used in scikit learn

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score
from numpy import mean

iris = load_iris()

X, y = iris.data, iris.target

#recursive binary splitting procedure needs a stopping criteria 

#min_samples_leaf and max_depth are used to prevent overfiting
#Pruning is another technique to prevent overfiting and simplify the model
clf = tree.DecisionTreeClassifier() #scikit-learn uses an optimised version of the CART (does not support categorical variables for now)

#train
clf = clf.fit(X, y)

#visualize
tree.plot_tree(clf)

#classify
scores = cross_val_score(clf, X, y)
print('Accuaracy: %.3f' % (mean(scores)))