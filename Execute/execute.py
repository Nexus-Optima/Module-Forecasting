import io
import os

import pandas as pd
import boto3
from dotenv import load_dotenv

# Importing necessary constants and parameters.
import Constants.constants as cts
import Constants.parameters as prms

# Utilities for data processing.
from Utils.process_data import process_data_lagged_rolling_stats, process_data_lagged

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

# Importing financial loss version 2
from Algorithm.financial_loss_2 import execute_purchase_strategy_v2
from Algorithm.financial_loss import execute_purchase_strategy


def forecast_pipeline():
    """Running the forecasting pipeling"""

    'Reading and Processing the data'
    read_df = read_data()
    processed_data = process_data_lagged(read_df, prms.FORECASTING_DAYS)

    'FEATURE ENGINEERING & SELECTION'
    features_dataset = create_features_dataset(processed_data.copy())
    features_dataset = features_dataset.last('4Y')

    'TUNE HYPER-PARAMETERS'
    params, actual_data, predictions = tune_xgboost_hyperparameters(features_dataset)
    lstm_params = tune_lstm_hyperparameters(features_dataset.copy(), no_trials=100)
    lstm_params = prms.lstm_parameters_4Y

    'EXECUTE MODELS'
    sarimax_forecast = execute_sarimax(features_dataset.copy(), prms.FORECASTING_DAYS)
    prophet_forecast = execute_prophet(features_dataset.copy(), prms.FORECASTING_DAYS)
    ets_predictions = execute_ets(features_dataset.copy(), prms.FORECASTING_DAYS)
    lstm_forecast = execute_lstm(read_df, features_dataset.copy(), prms.FORECASTING_DAYS, lstm_params)
    actual_data, predictions, future_data = \
        execute_adaptive_xgboost(read_df, features_dataset.copy(), prms.FORECASTING_DAYS, prms.xgboost_params_4Y)
    lgbm_predictions, lgbm_forecast = execute_lgbm(processed_data.copy(), prms.FORECASTING_DAYS)

    'EXECUTE PURCHASE STRATEGY'
    execute_purchase_strategy(lgbm_predictions, actual_data, 10, 0, 400)
    execute_purchase_strategy_v2(read_df, features_dataset.copy(), 12833, 40, prms.FORECASTING_DAYS)


def create_features_dataset(processed_data):
    """Function to create a dataset using selected features based on correlation methods."""
    common_selected_features = set(processed_data.columns)
    for method_name in cts.correlation_methods:
        selected_features_df = method_name(processed_data)
        common_selected_features = common_selected_features.intersection(set(selected_features_df.columns))
    common_selected_features.discard('Output')
    features_dataset = processed_data[['Output'] + list(common_selected_features)]

    return features_dataset


def standardize_dataset(df, date_column, num_columns):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    for col in num_columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

    return df


def read_data():
    """Function to read and process the data."""

    def custom_date_parser(date_string):
        return pd.to_datetime(date_string, format='%d/%m/%y')

    # data = pd.read_csv('../Data/Price_Data.csv', parse_dates=['Date'], date_parser=custom_date_parser)
    files = ['../Data/Updated_Data_Cotton/Brazil Cotton Index.csv',
             '../Data/Updated_Data_Cotton/China Cotton Futures.csv',
             '../Data/Updated_Data_Cotton/Cotlook.csv', '../Data/Updated_Data_Cotton/IND Cotton Futures.csv',
             '../Data/Updated_Data_Cotton/Spot Prices.csv', '../Data/Updated_Data_Cotton/US Cotton Futures.csv']

    brazil_cotton_index = pd.read_csv('../Data/Updated_Data_Cotton/Brazil Cotton Index.csv', parse_dates=['Date'],
                                      dayfirst=True)
    china_cotton_index = pd.read_csv('../Data/Updated_Data_Cotton/China Cotton Futures.csv', parse_dates=['Date'],
                                     dayfirst=True)
    cotlook_index = pd.read_csv('../Data/Updated_Data_Cotton/Cotlook.csv', parse_dates=['Date'],
                                dayfirst=True)
    ind_cotton_index = pd.read_csv('../Data/Updated_Data_Cotton/IND Cotton Futures.csv', parse_dates=['Date'],
                                   dayfirst=True)
    spot_prices = pd.read_csv('../Data/Updated_Data_Cotton/Spot Prices.csv', parse_dates=['Date'],
                              dayfirst=True)
    us_cotton_index = pd.read_csv('../Data/Updated_Data_Cotton/US Cotton Futures.csv', parse_dates=['Date'],
                                  dayfirst=True)

    datasets = [brazil_cotton_index, cotlook_index, ind_cotton_index, spot_prices, china_cotton_index,
                us_cotton_index]

    standardized_datasets = []

    for df in datasets:
        standardized_df = standardize_dataset(df, 'Date', df.columns.drop('Date'))
        standardized_datasets.append(standardized_df)

    all_dates = pd.date_range(start=spot_prices['Date'].min(),
                              end=max(df['Date'].max() for df in standardized_datasets))
    date_df = pd.DataFrame(all_dates, columns=['Date'])
    for df in standardized_datasets:
        date_df = pd.merge(date_df, df, on='Date', how='left')

    date_column = date_df['Date']
    date_df.drop('Date', axis=1, inplace=True)
    date_df.interpolate(method='linear', inplace=True)
    date_df['Date'] = date_column
    date_df = date_df[['Date'] + [col for col in date_df.columns if col != 'Date']]

    # processed_data = process_data_lagged(data, prms.FORECASTING_DAYS)
    # # cols_to_remove = (set(processed_data.columns) & set(data.columns)) - {"Output"}
    # # processed_data = processed_data.drop(columns=cols_to_remove)
    # test = processed_data['Output'][int(0.8 * len(processed_data)):]

    return date_df


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


# load_dotenv()
forecast_pipeline()
