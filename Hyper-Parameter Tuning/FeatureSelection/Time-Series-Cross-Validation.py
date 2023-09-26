import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

data_path = "../../Data/ICAC multiple variables.csv"  # replace with your data path
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

data = data.resample('D').mean().fillna(method='ffill')

subset_data = data.last('1Y')
initial_train_size = int(0.7 * len(subset_data))
step_size = 7
actual_values = []
predictions = []

for train_end_idx in range(initial_train_size, len(subset_data) - step_size, step_size):
    train_data = subset_data.iloc[:train_end_idx]
    val_data = subset_data.iloc[train_end_idx:train_end_idx + step_size]

    X_train, y_train = train_data.drop(columns='Output'), train_data['Output']
    X_val, y_val = val_data.drop(columns='Output'), val_data['Output']

    model = xgb.XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    actual_values.extend(y_val.values)
    predictions.extend(prediction)

mae = mean_absolute_error(actual_values, predictions)
print("Mean Absolute Error:", mae)
