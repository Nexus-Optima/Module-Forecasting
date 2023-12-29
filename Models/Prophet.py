
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def execute_prophet(data, forecast_days):
    data = data.reset_index()
    data.rename(columns={"Date": "ds", "Output": "y"}, inplace=True)

    test_size = int(0.2 * len(data))
    train_size = len(data) - test_size
    train, test = data[:train_size], data[train_size:]

    model = Prophet()
    model.fit(train)
    future_dates = model.make_future_dataframe(periods=test_size)
    forecast = model.predict(future_dates[-test_size:])

    model_future = Prophet()
    model_future.fit(data)
    future_dates_final = model_future.make_future_dataframe(periods=forecast_days)
    forecast_final = model_future.predict(future_dates_final[-forecast_days:])

    plt.figure(figsize=(14, 7))

    plt.plot(train['ds'], train['y'], label="Training Data", color="blue")
    plt.plot(test['ds'], test['y'], label="Test Data", color="green")
    plt.plot(forecast['ds'], forecast['yhat'], 'r--', label="Predicted Test Data", alpha=0.7)
    plt.plot(forecast_final['ds'], forecast_final['yhat'], 'y', label="100-day Forecast", alpha=0.7)

    plt.legend()
    plt.title("Time Series Forecasting with Prophet")
    plt.show()

    rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat'].head(test_size)))
    print(f"RMSE: {rmse}")
    return forecast_final