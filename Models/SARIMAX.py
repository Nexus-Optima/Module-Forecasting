import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../Data/ICAC Data.csv', parse_dates=['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

lags = [1, 2, 3, 7, 14, 21, 28]
for lag in lags:
    df[f"lag_{lag}"] = df['Cotlook_A_index'].shift(lag)

window_sizes = [7, 14, 21, 28]
for window in window_sizes:
    df[f"rolling_mean_{window}"] = df['Cotlook_A_index'].rolling(window=window).mean()
    df[f"rolling_std_{window}"] = df['Cotlook_A_index'].rolling(window=window).std()

df_fe = df.dropna()

train_size = int(0.8 * len(df_fe))
train, test = df_fe.iloc[:train_size], df_fe.iloc[train_size:]

X_train, X_test = train.drop("Cotlook_A_index", axis=1), test.drop("Cotlook_A_index", axis=1)
y_train, y_test = train["Cotlook_A_index"], test["Cotlook_A_index"]

sarimax_model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
sarimax_result = sarimax_model.fit(disp=False)

sarimax_predictions = sarimax_result.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)

sarimax_rmse = np.sqrt(mean_squared_error(y_test, sarimax_predictions))
print(f"RMSE for SARIMAX model: {sarimax_rmse:.4f}")

forecast_df = df_fe.iloc[-len(y_test):].copy()
exog_forecast = forecast_df[X_train.columns]
sarimax_forecast_45 = sarimax_result.get_forecast(steps=45, exog=exog_forecast).predicted_mean

print(sarimax_forecast_45)

plt.figure(figsize=(16, 8))
plt.plot(df['Cotlook_A_index'], label="Actual Data", color="blue")
plt.plot(y_test.index, sarimax_predictions, label="SARIMAX Predictions", color="red", linestyle="--")
forecast_index_45 = pd.date_range(df.index[-1], periods=46, inclusive='right')
plt.plot(forecast_index_45, sarimax_forecast_45, 'g--', label="45-day SARIMAX Forecast", alpha=0.7)
plt.title("Cotlook A Index with SARIMAX Predictions and 45 Days Forecast")
plt.xlabel("Date")
plt.ylabel("Cotlook A Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
