#new2.py
from sklearn.model_selection import KFold
import pandas as pd

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
x = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]

kf = KFold(n_splits=3, shuffle=True)
splits = list(kf.split(x))
first_split = splits[0]
train_indices, test_indices = first_split
print("train set indices:", train_indices)
print("test set indices:", test_indices)

x_train = x[train_indices]
x_test = x[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]
print("x_train:")
print(x_train)
print("y_train:", y_train)
print(" x_test:")
print(x_test)
print(" y_test:", y_test)