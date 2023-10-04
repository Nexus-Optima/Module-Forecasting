import pandas as pd
from Utils.process_data import process_data, process_data_lagged_rolling_stats, process_data_lagged
from Models.Adaptive_XGBoost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima

from Feature_Selection.Time_Series_Decomposition import evaluate_time_series_decomposition
from Feature_Selection.Correlation_Analysis import evaluate_correlation_analysis
from Feature_Selection.Tree_Based_Models import evaluate_tree_based_models
from Feature_Selection.Recursive_Feature_Elimination import evaluate_recursive_feature_elimination
from Feature_Selection.Spearman_Correlation import evaluate_spearman_correlation_analysis
from DL_Models.LSTM import execute_lstm

data = pd.read_csv("../Data/ICAC multiple variables.csv", parse_dates=['Date'], dayfirst=True)
processed_data = process_data(data)


def execute():
    adaptive_xgboost_result = execute_adaptive_xgboost(processed_data)
    ets_result = execute_ets(processed_data)
    arima_result = execute_arima(processed_data)
    lstm_result = execute_lstm(processed_data)


def evaluate():
    evaluate_time_series_decomposition(processed_data)
    evaluate_correlation_analysis(processed_data)
    evaluate_tree_based_models(processed_data)
    evaluate_spearman_correlation_analysis(processed_data)
    evaluate_recursive_feature_elimination(processed_data)


evaluate()
execute()
