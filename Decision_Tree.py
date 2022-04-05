

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score
from numpy import mean

iris = load_iris()

X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)

scores = cross_val_score(clf, X, y)
print('Accuaracy: %.3f' % (mean(scores)))