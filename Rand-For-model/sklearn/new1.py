import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

x = df[cancer_data.feature_names].values
y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
first_row = x_test[0]

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("prediction:", rf.predict([first_row]))
print("prediction using Decision Tree:", dt.predict([first_row]))
print("true value:", y_test[0])
print("random forest accuracy:", rf.score(x_test, y_test))
print("Decision tree accuracy:", dt.score(x_test, y_test))