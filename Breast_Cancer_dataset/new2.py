#new2.py
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])


df['target'] = cancer_data['target']

print(cancer_data['target'])
print(cancer_data['target'].shape)
print(cancer_data['target_names'])
print(df.head())