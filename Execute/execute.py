from Algorithm.financial_loss import execute_purchase_strategy

import pandas as pd
from Utils.process_data import process_data, process_data_lagged_rolling_stats, process_data_lagged
from Models.Adaptive_XGBoost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima
from Models.Prophet import execute_prophet
from Models.LightGBM import execute_lgbm

from Feature_Selection.Time_Series_Decomposition import evaluate_time_series_decomposition
from Feature_Selection.Correlation_Analysis import evaluate_correlation_analysis
from Feature_Selection.Tree_Based_Models import evaluate_tree_based_models
from Feature_Selection.Recursive_Feature_Elimination import evaluate_recursive_feature_elimination
from Feature_Selection.Spearman_Correlation import evaluate_spearman_correlation_analysis
from DL_Models.LSTM import execute_lstm

data = pd.read_csv("../Data/ICAC multiple variables.csv", dayfirst=True)
processed_data = process_data_lagged(data)
train_size = int(0.8 * len(processed_data))
test = processed_data['Output'][train_size:]


def execute_models():
    execute_ets(processed_data)
    execute_arima(processed_data)
    execute_lstm(processed_data)
    execute_adaptive_xgboost(processed_data)
    execute_adaptive_xgboost(processed_data)
    execute_prophet(processed_data)
    predicted_prices = execute_lgbm(processed_data)

    # Execute Purchase Strategy
    execute_purchase_strategy(predicted_prices, test, 10, 0, 400)


def evaluate_models():
    evaluate_time_series_decomposition(processed_data)
    evaluate_correlation_analysis(processed_data)
    evaluate_tree_based_models(processed_data)
    evaluate_spearman_correlation_analysis(processed_data)
    evaluate_recursive_feature_elimination(processed_data)


evaluate_models()
execute_models()
