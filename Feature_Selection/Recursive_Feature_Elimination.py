from sklearn.feature_selection import RFE
import pandas as pd
import xgboost as xgb

def evaluate_recursive_feature_elimination(data):

    X = data.drop(columns='Output')
    y = data['Output']
    xgb_model_quick = xgb.XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, objective='reg:squarederror', random_state=42)
    xgb_model_quick.fit(X, y)
    selector = RFE(xgb_model_quick, n_features_to_select=10, step=1)
    selector = selector.fit(X, y)
    
    feature_ranking = selector.ranking_
    rfe_ranking_df = pd.DataFrame({
        'Feature': X.columns,
        'RFE_Rank': feature_ranking
    }).sort_values(by='RFE_Rank')
    
    print(rfe_ranking_df)