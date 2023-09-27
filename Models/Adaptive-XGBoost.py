import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from Evaluation.XGBoostEvaluation import getPredictions

data = pd.read_csv("../Data/ICAC multiple variables.csv", parse_dates=['Date'], dayfirst=True)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)
data = data.resample('D').mean().fillna(method='ffill')
subset_data = data.last('2Y')

window_size = int(0.5 * len(subset_data))
actual_values, predictions = getPredictions()

train_data = subset_data[-window_size:]
X_train, y_train = train_data.drop(columns='Output'), train_data['Output']

model_future = xgb.XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, objective='reg:squarederror', random_state=42)
model_future.fit(X_train, y_train)

future_dates = [subset_data.index[-1] + pd.Timedelta(days=i) for i in range(1, 46)]
future_data = pd.DataFrame(index=future_dates)
future_data['Day'] = [date.day for date in future_dates]
future_data['Month'] = [date.month for date in future_dates]
future_data['Year'] = [date.year for date in future_dates]

for col in data.columns:
    if col not in ['Day', 'Month', 'Year', 'Output']:
        future_data[col] = data[col].iloc[-1]
future_data = future_data[X_train.columns]
predicted_output = []

for i in range(45):
    prediction = model_future.predict(future_data.iloc[[i]].drop(columns='Output', errors='ignore'))
    future_data.loc[future_data.index[i], 'Output'] = prediction[0]
    predicted_output.append(prediction[0])

plt.figure(figsize=(14, 7))
plt.plot(subset_data.index[window_size:window_size+len(predictions)], actual_values, label="Actual Values", color='blue')
plt.plot(subset_data.index[window_size:window_size+len(predictions)], predictions, label="Predicted Values", color='red', linestyle='--')
plt.plot(future_data.index, future_data['Output'], label="Forecast for Next 45 Days", color='green', linestyle='-.')

plt.title("Actual, Predicted and Forecasted Values")
plt.xlabel("Date")
plt.ylabel("Output")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()