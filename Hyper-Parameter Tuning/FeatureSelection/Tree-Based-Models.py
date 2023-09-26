from sklearn.ensemble import RandomForestRegressor
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


rf = RandomForestRegressor(n_estimators=100, random_state=20)
rf.fit(X, y)

feature_importances = rf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', color='skyblue')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
