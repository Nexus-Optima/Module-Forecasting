from dotenv import load_dotenv

# Importing necessary constants and parameters.
import Constants.constants as cts
import Constants.parameters as prms
from Data_Preprocessing.data_processing import create_features_dataset
from Database.s3_operations import read_data_s3, store_forecast

# Utilities for data processing.
from Utils.process_data import process_data_lagged

# Importing different modeling approaches.
from Models.XG_Boost.adaptive_xgboost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.SARIMAX import execute_sarimax

from DL_Models.LSTM.LSTM import execute_lstm

# Importing tuning functions for XGBoost and LSTM.
from Models.XG_Boost.xgboost_tuning import tune_xgboost_hyperparameters
from DL_Models.LSTM.lstm_tuning import tune_lstm_hyperparameters

# Importing financial loss version 2
from Algorithm.financial_loss_2 import execute_purchase_strategy_v2
from Algorithm.financial_loss import execute_purchase_strategy


def forecast_pipeline(commodity_name):
    load_dotenv()
    """Running the forecasting pipeline"""

    'Reading and Processing the data'
    read_df = read_data_s3(cts.Commodities.COMMODITIES, commodity_name)
    processed_data = process_data_lagged(read_df, prms.FORECASTING_DAYS)

    'FEATURE ENGINEERING & SELECTION'
    features_dataset = create_features_dataset(processed_data.copy())
    features_dataset = features_dataset.last('4Y')

    'TUNE HYPER-PARAMETERS'
    # params, actual_data, predictions = tune_xgboost_hyperparameters(features_dataset)
    # lstm_params = tune_lstm_hyperparameters(features_dataset.copy(), no_trials=1000)
    lstm_params = prms.lstm_parameters_4Y_30D

    'EXECUTE MODELS'
    # sarimax_forecast = execute_sarimax(features_dataset.copy(), prms.FORECASTING_DAYS)
    # ets_predictions = execute_ets(features_dataset.copy(), prms.FORECASTING_DAYS)
    test_predictions_orig, y_test_orig, forecast_orig = execute_lstm(read_df, features_dataset.copy(),
                                                                     prms.FORECASTING_DAYS, lstm_params)
    datasets = {"actual_values": y_test_orig, "forecast_values": forecast_orig}
    store_forecast(cts.Commodities.FORECAST_STORAGE, commodity_name, datasets)

    # actual_data, predictions, future_data = \
    # execute_adaptive_xgboost(read_df, features_dataset.copy(), prms.FORECASTING_DAYS, prms.xgboost_params_4Y)

    'EXECUTE PURCHASE STRATEGY'
    # execute_purchase_strategy(lgbm_predictions, actual_data, 10, 0, 400)
    # execute_purchase_strategy_v2(read_df, features_dataset.copy(), 12833, 40, prms.FORECASTING_DAYS)
