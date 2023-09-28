import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../../Data/ICAC multiple variables.csv')
data['Date'] = pd.to_datetime(data['Date'])
data_ts = data.set_index('Date')

data_ts_grouped = data_ts.groupby(data_ts.index).mean()
data_ts_filled = data_ts_grouped.resample('D').ffill()
X = data_ts_filled.drop(columns='Output')
y = data_ts_filled['Output']

xgb_model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
xgb_model.fit(X, y)

xgb_importances = xgb_model.feature_importances_

xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'XGB_Importance': xgb_importances
}).sort_values(by='XGB_Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=xgb_importance_df, x='XGB_Importance', y='Feature', color='lightgreen')
plt.title('Feature Importance from XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
