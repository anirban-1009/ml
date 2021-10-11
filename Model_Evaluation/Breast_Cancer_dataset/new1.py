import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
print(cancer_data['data'].shape)
print(cancer_data['feature_names'])
print(df.head())