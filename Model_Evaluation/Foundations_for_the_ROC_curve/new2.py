#new2.py


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred_proba1 = model.predict_proba(x_test)
print("model 1  AUC score:", roc_auc_score(y_test, y_pred_proba1[:, 1]))

model2 = LogisticRegression()
model2.fit(x_train[:, 0:2], y_train)
y_pred_proba2 = model2.predict_proba(x_test[:, 0:2])
print("model 1 AUC score:", roc_auc_score(y_test, y_pred_proba2[:, 1]))