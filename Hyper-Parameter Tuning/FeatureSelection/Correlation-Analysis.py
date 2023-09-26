import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../../Data/ICAC multiple variables.csv')
correlations_no_date = data.drop(columns='Date').corr()['Output'].drop('Output')
plt.figure(figsize=(12, 6))
correlations_no_date.sort_values().plot(kind='bar', color='skyblue')
plt.title('Feature Correlation with Output (Excluding Date)')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
plt.tight_layout()
plt.show()
