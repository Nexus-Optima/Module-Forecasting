from Feature_Selection.Correlation_Analysis import evaluate_correlation_analysis
from Feature_Selection.Spearman_Correlation import evaluate_spearman_correlation_analysis

from Models.XG_Boost.adaptive_xgboost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima
from Models.Prophet import execute_prophet
from Models.LightGBM import execute_lgbm

from DL_Models.LSTM.LSTM import execute_lstm

correlation_methods = [evaluate_correlation_analysis, evaluate_spearman_correlation_analysis]

forecasting_models = [execute_adaptive_xgboost, execute_arima, execute_ets, execute_prophet, execute_lgbm,
                      execute_lstm]