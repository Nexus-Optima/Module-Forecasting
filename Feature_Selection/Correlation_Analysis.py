import matplotlib.pyplot as plt
import seaborn as sns

threshold = 0.5


def evaluate_correlation_analysis(data):
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

    selected_features = correlation_with_output[correlation_with_output.abs() > threshold].index.tolist()
    selected_df = data[['Output'] + selected_features]

    return selected_df
