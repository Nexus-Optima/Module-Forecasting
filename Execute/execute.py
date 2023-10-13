from Algorithm.financial_loss import execute_purchase_strategy

import Constants.constants as cts

import pandas as pd
from Utils.process_data import process_data, process_data_lagged_rolling_stats, process_data_lagged
from Models.Adaptive_XGBoost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima
from Models.Prophet import execute_prophet
from Models.LightGBM import execute_lgbm

from DL_Models.LSTM import execute_lstm

from Evaluation.adaptive_xgboost_evaluation import getPredictions


def forecast_pipeline():
    processed_data, actual_data = read_data()
    features_dataset = create_features_dataset(processed_data.copy())

    execute_ets(processed_data.copy())
    execute_arima(processed_data.copy())
    execute_lstm(processed_data.copy())
    actual_prices, predicted_prices = getPredictions(processed_data.copy())
    execute_adaptive_xgboost(processed_data.copy())
    execute_prophet(processed_data.copy())
    predicted_prices = execute_lgbm(processed_data.copy())

    # Execute Purchase Strategy
    execute_purchase_strategy(predicted_prices, actual_data, 10, 0, 400)


def create_features_dataset(processed_data):
    common_selected_features = set(processed_data.columns)
    for method_name in cts.correlation_methods:
        selected_features_df = method_name(processed_data)
        common_selected_features = common_selected_features.intersection(set(selected_features_df.columns))
    common_selected_features.discard('Output')
    features_dataset = processed_data[['Output'] + list(common_selected_features)]

    return features_dataset


def read_data():
    data = pd.read_csv("../Data/ICAC multiple variables.csv", dayfirst=True)
    processed_data = process_data_lagged(data)
    test = processed_data['Output'][int(0.8 * len(processed_data)):]

    return processed_data, test


forecast_pipeline()
