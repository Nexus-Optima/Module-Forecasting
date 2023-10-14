from Feature_Selection.Correlation_Analysis import evaluate_correlation_analysis
from Feature_Selection.Recursive_Feature_Elimination import evaluate_recursive_feature_elimination
from Feature_Selection.Spearman_Correlation import evaluate_spearman_correlation_analysis
from Feature_Selection.Tree_Based_Models import evaluate_tree_based_models

from Models.Adaptive_XGBoost import execute_adaptive_xgboost
from Models.ETS import execute_ets
from Models.Arima import execute_arima
from Models.Prophet import execute_prophet
from Models.LightGBM import execute_lgbm

from DL_Models.LSTM import execute_lstm

correlation_methods = [evaluate_correlation_analysis, evaluate_spearman_correlation_analysis,
                       evaluate_recursive_feature_elimination]

forecasting_models = [execute_adaptive_xgboost, execute_arima, execute_ets, execute_prophet, execute_lgbm,
                      execute_lstm]