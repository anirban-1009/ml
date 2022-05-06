"""new3.py"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

x = df[['Pclass', 'male','Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(x, y)

y_pred = model.predict(x)
print((y == y_pred).sum())
print((y == y_pred).sum() / y.shape[0])
print(model.score(x, y))
