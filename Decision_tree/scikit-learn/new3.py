from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

features_names = ['Pclass', 'male']
df['male'] = df['Sex'] == 'male'
x = df[['Pclass', 'male']].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(x, y)

dot_file = export_graphviz(dt, feature_names=features_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)