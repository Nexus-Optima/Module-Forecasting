import io
import os

import pandas as pd
import boto3
from dotenv import load_dotenv

# Importing necessary constants and parameters.
import Constants.constants as cts
import Constants.parameters as prms

# Utilities for data processing.
from Utils.process_data import process_data_lagged_rolling_stats

# Importing different modeling approaches.
from Models.XG_Boost.adaptive_xgboost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Prophet import execute_prophet
from Models.SARIMAX import execute_sarimax
from Models.LightGBM import execute_lgbm

from DL_Models.LSTM.LSTM import execute_lstm

# Importing tuning functions for XGBoost and LSTM.
from Models.XG_Boost.xgboost_tuning import tune_xgboost_hyperparameters
from DL_Models.LSTM.lstm_tuning import tune_lstm_hyperparameters


def forecast_pipeline():
    """Running the forecasting pipeling"""

    'Reading and Processing the data'
    processed_data, actual_data = read_data()

    'FEATURE ENGINEERING & SELECTION'
    features_dataset = create_features_dataset(processed_data.copy())
    features_dataset.last('4Y')

    'TUNE HYPER-PARAMETERS'
    params, actual_data, predictions = tune_xgboost_hyperparameters(features_dataset)
    lstm_params = tune_lstm_hyperparameters(features_dataset.copy(), no_trials=100)

    'EXECUTE MODELS'
    sarimax_forecast = execute_sarimax(features_dataset.copy(), prms.FORECASTING_DAYS)
    prophet_forecast = execute_prophet(features_dataset.copy(), prms.FORECASTING_DAYS)
    ets_predictions = execute_ets(features_dataset.copy(), prms.FORECASTING_DAYS)
    lstm_forecast = execute_lstm(features_dataset.copy(), prms.FORECASTING_DAYS, lstm_params)
    xgboost_predictions, xgboost_forecast = execute_adaptive_xgboost(features_dataset.copy(), prms.FORECASTING_DAYS,
                                                                     prms.xgboost_params_4Y)
    lgbm_predictions, lgbm_forecast = execute_lgbm(processed_data.copy(), prms.FORECASTING_DAYS)

    'EXECUTE PURCHASE STRATEGY'
    # execute_purchase_strategy(lgbm_predictions, actual_data, 10, 0, 400)


def create_features_dataset(processed_data):
    """Function to create a dataset using selected features based on correlation methods."""
    common_selected_features = set(processed_data.columns)
    for method_name in cts.correlation_methods:
        selected_features_df = method_name(processed_data)
        common_selected_features = common_selected_features.intersection(set(selected_features_df.columns))
    common_selected_features.discard('Output')
    features_dataset = processed_data[['Output'] + list(common_selected_features)]

    return features_dataset


def read_data():
    """Function to read and process the data."""

    def custom_date_parser(date_string):
        return pd.to_datetime(date_string, format='%m/%d/%y')

    data = pd.read_csv('../Data/Price_Data.csv', parse_dates=['Date'], date_parser=custom_date_parser)
    processed_data = process_data_lagged_rolling_stats(data, prms.FORECASTING_DAYS)
    cols_to_remove = (set(processed_data.columns) & set(data.columns)) - {"Output"}
    processed_data = processed_data.drop(columns=cols_to_remove)
    test = processed_data['Output'][int(0.8 * len(processed_data)):]

    return processed_data, test


def read_data_s3(BUCKET_NAME, FILE_NAME):
    """Function to read and process the data from an S3 bucket."""
    def custom_date_parser(date_string):
        return pd.to_datetime(date_string, format='%m/%d/%y')

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    csv_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=FILE_NAME)
    body = csv_obj['Body']

    data = pd.read_csv(io.BytesIO(body.read()), parse_dates=['Date'], date_parser=custom_date_parser)

    processed_data = process_data_lagged_rolling_stats(data, prms.FORECASTING_DAYS)
    cols_to_remove = (set(processed_data.columns) & set(data.columns)) - {"Output"}
    processed_data = processed_data.drop(columns=cols_to_remove)
    test = processed_data['Output'][int(0.8 * len(processed_data)):]

    return processed_data, test


load_dotenv()
forecast_pipeline()
