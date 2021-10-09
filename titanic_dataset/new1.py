#new1.py
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
x = df[['Fare', 'Age']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(x, y)
print(model.coef_, model.intercept_)