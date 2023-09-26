from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_arima_model(train, test, order):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    return rmse

df = pd.read_csv('../Data/ICAC Data.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df.head()
train_size = int(0.8 * len(df))
train, test = df['Output'][:train_size], df['Output'][train_size:]
best_rmse = float('inf')
best_order = None
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                rmse = evaluate_arima_model(train, test, order)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
            except:
                continue

model = ARIMA(df['Output'], order=best_order)
model_fit = model.fit()
forecast = model_fit.forecast(steps=100)
print(forecast)

plt.figure(figsize=(14, 7))
plt.plot(df['Output'], label="Actual Data")
forecast_index = pd.date_range(df.index[-1], periods=101, inclusive='right')
plt.plot(forecast_index, forecast, 'r--', label="Forecast")
plt.title("Cotlook A Index Forecast")
plt.xlabel("Date")
plt.ylabel("Cotlook A Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(best_rmse)
print(best_order)