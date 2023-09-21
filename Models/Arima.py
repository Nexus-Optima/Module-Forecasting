import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load your time series data into a pandas DataFrame
# Replace 'your_data.csv' with your actual data file or source
data = pd.read_csv('../Data/ICAC Data.csv')
# Assuming you have a 'date' column in your DataFrame, convert it to a datetime object
data['Date'] = pd.to_datetime(data['Date'])
# Set the 'date' column as the index
data.set_index('Date', inplace=True)

# Plot the time series data to visualize it
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Cotlook_A_Index')
plt.show()

# Check for stationarity using the Augmented Dickey-Fuller test
result = adfuller(data['Cotlook_A_index'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')

# If the time series is not stationary, perform differencing
if result[1] > 0.05:
    data_diff = data.diff().dropna()
else:
    data_diff = data

# Plot ACF and PACF to determine the order of AR and MA components
plt.figure(figsize=(12, 6))
plot_acf(data_diff, lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data_diff, lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Fit an ARIMA model
p = 1  # Order of AR
d = 1  # Degree of differencing
q = 1  # Order of MA

model = sm.tsa.ARIMA(data_diff, order=(p, d, q))
results = model.fit()

# Print the model summary
print(results.summary())

# Plot the residuals
residuals = results.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.show()

# Make predictions with the ARIMA model
forecast_steps = 10  # Number of steps to forecast into the future
forecast = results.forecast(steps=forecast_steps)

# Create a date range for the forecasted values
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, closed='right')

# Convert forecast values to a DataFrame
forecast_df = pd.DataFrame(data=forecast, index=forecast_index, columns=['Forecast'])

# Plot the original data and the forecast
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original Data')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title('Original Data vs. Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
