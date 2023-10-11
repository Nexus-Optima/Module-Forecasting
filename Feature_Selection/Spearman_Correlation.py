import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Utils.process_data import process_data_lagged


def evaluate_spearman_correlation_analysis(data):
    spearman_correlations = data.corr(method='spearman')['Output'].sort_values(ascending=False)
    spearman_correlations.drop('Output', inplace=True)
    print(spearman_correlations)
    spearman_correlations.sort_values().plot(kind='barh', figsize=(10, 8))
    plt.title('Correlation of Features with Output')
    plt.xlabel('Correlation Coefficient')
    plt.show()
