import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

df = pd.read_csv('../Data/ICAC Data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

Q1 = df['Cotlook_A_index'].quantile(0.25)
Q3 = df['Cotlook_A_index'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Cotlook_A_index'] = df['Cotlook_A_index'].where((df['Cotlook_A_index'] >= lower_bound) &
                                                    (df['Cotlook_A_index'] <= upper_bound))
df['Cotlook_A_index'].interpolate(method='linear', inplace=True)

train_size = int(0.8 * len(df))
train, test = df['Cotlook_A_index'][:train_size], df['Cotlook_A_index'][train_size:]

model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=365, damped_trend=True)
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, predictions))

forecast_next_100_days = model_fit.forecast(steps=100)

plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label="Training Data", color="blue")
plt.plot(test.index, test, label="Test Data", color="green")
plt.plot(test.index, predictions, label="Adjusted ETS Predictions", color="red", linestyle="--")
forecast_index_100 = pd.date_range(test.index[-1], periods=101, inclusive='right')
plt.plot(forecast_index_100, forecast_next_100_days, 'r--', label="100-day Forecast (Adjusted)", alpha=0.7)
plt.title("Cotlook A Index with Adjusted ETS Model Predictions and 100 Days Forecast")
plt.xlabel("Date")
plt.ylabel("Cotlook A Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
