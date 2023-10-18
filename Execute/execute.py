from Algorithm.financial_loss import execute_purchase_strategy

import Constants.constants as cts
from Constants.parameters import xgboost_params

import pandas as pd
from Utils.process_data import process_data, process_data_lagged_rolling_stats, process_data_lagged
from Models.Adaptive_XGBoost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima
from Models.Prophet import execute_prophet
from Models.LightGBM import execute_lgbm
from Models.SARIMAX import execute_sarimax

from Parameter_Tuning.tuning import tune_hyperparameters

from DL_Models.LSTM import execute_lstm


def forecast_pipeline():
    processed_data, actual_data = read_data()
    features_dataset = create_features_dataset(processed_data.copy())
    features_dataset.last('4Y')

    # execute_ets(processed_data.copy())
    # execute_arima(processed_data.copy())
    # execute_lstm(features_dataset.copy())
    # execute_sarimax(features_dataset.copy())
    execute_adaptive_xgboost(features_dataset.copy(), 15, xgboost_params)
    # execute_prophet(processed_data.copy())
    # predicted_prices = execute_lgbm(processed_data.copy())

    # Execute Purchase Strategy
    # execute_purchase_strategy(predicted_prices, actual_data, 10, 0, 400)


def create_features_dataset(processed_data):
    common_selected_features = set(processed_data.columns)
    for method_name in cts.correlation_methods:
        selected_features_df = method_name(processed_data)
        common_selected_features = common_selected_features.intersection(set(selected_features_df.columns))
    common_selected_features.discard('Output')
    features_dataset = processed_data[['Output'] + list(common_selected_features)]

    return features_dataset


def read_data():
    def custom_date_parser(date_string):
        return pd.to_datetime(date_string, format='%m/%d/%y')

    data = pd.read_csv('../Data/Price_Data.csv', parse_dates=['Date'], date_parser=custom_date_parser)
    processed_data = process_data_lagged_rolling_stats(data, 20)
    cols_to_remove = (set(processed_data.columns) & set(data.columns)) - {"Output"}
    processed_data = processed_data.drop(columns=cols_to_remove)
    test = processed_data['Output'][int(0.8 * len(processed_data)):]

    return processed_data, test


forecast_pipeline()
