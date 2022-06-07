#one hot encoding (scikit learn) = dummy encoding (panda)

# dummy encoding of categorical features
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import make_column_transformer

df = pd.read_csv('http://bit.ly/kaggletrain')
df.shape
df.isna().sum()

#leave only 4 features
df = df.loc[df.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Embarked']]

#convert sex 
ohe = OneHotEncoder(sparse=False)
#ohe.fit_transform(df[['Sex']])



X = df.drop('Survived', axis='columns')
y = df.Survived

column_trans = make_column_transformer(
    (OneHotEncoder(), ['Sex', 'Embarked']),
    remainder='passthrough')

#testing
logreg = LogisticRegression(solver='lbfgs')
pipe = make_pipeline(column_trans, logreg)

cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

print("new data:")
print(column_trans.fit_transform(df[['Sex', 'Embarked']]))



