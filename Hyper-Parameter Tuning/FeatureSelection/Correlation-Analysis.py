import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Utils.process_data import process_data_lagged

data = pd.read_csv("../../Data/ICAC multiple variables.csv", parse_dates=['Date'], dayfirst=True)
data = process_data_lagged(data)
correlation_matrix = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Analysis")
plt.show()

correlation_with_output = data.corr()['Output']
correlation_with_output.drop('Output', inplace=True)
correlation_with_output.sort_values().plot(kind='barh', figsize=(10, 8))
plt.title('Correlation of Features with Output')
plt.xlabel('Correlation Coefficient')
plt.show()