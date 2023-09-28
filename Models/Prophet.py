from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from prophet.diagnostics import cross_validation

df = pd.read_csv('../Data/ICAC Data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date')
df.sort_index()
print(df.head())
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

test_size = int(0.2 * len(df))
train_size = len(df) - test_size
test, train = df[:test_size], df[test_size:]

model = Prophet()
model.fit(train)
modelFuture = Prophet()
modelFuture.fit(df)
future_dates = model.make_future_dataframe(periods=len(test))
future_dates = future_dates[train_size:]
future_dates_final = modelFuture.make_future_dataframe(periods=100)
forecast = model.predict(future_dates)
forecastFinal = modelFuture.predict(future_dates_final)
print(forecast.head())
print(forecastFinal.head())
print(test.head())
print(len(test))
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
plt.figure(figsize=(14, 7))

# model.plot(forecast['yhat'], uncertainty=True)
plt.plot(train['ds'], train['y'], label="Training Data", color="blue")
plt.plot(test['ds'], test['y'], label="Test Data", color="green")
plt.plot(forecast['ds'], forecast['yhat'], 'r--', label="Predicted Test Data", alpha=0.7)
plt.plot(forecastFinal['ds'], forecastFinal['yhat'], 'y', label="100-day Forecast (Adjusted)", alpha=0.7)
plt.legend()
model.plot_components(forecast)
modelFuture.plot_components((forecastFinal))
plt.show()
rmse = np.sqrt(mean_squared_error(test, forecast))
print(rmse)
