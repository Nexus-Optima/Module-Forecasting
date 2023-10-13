import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def executeEvaluation(subset_data):
    predictions = []
    actual_values = []
    window_size = int(0.5 * len(subset_data))
    # TODO: Function called time_series_split exists which can be used !!
    for window_start in range(0, len(subset_data) - window_size):
        train_data = subset_data.iloc[window_start:window_start + window_size]
        val_data = subset_data.iloc[window_start + window_size:window_start + window_size + 1]

        X_train, y_train = train_data.drop(columns='Output'), train_data['Output']
        X_val, y_val = val_data.drop(columns='Output'), val_data['Output']

        model = xgb.XGBRegressor(n_estimators=50, max_depth=5, n_jobs=-1, objective='reg:squarederror',
                                 random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

        prediction = model.predict(X_val)

        actual_values.extend(y_val.values)
        predictions.extend(prediction)

    plt.figure(figsize=(14, 7))
    plt.plot(subset_data.index[window_size:window_size + len(predictions)], actual_values, label="Actual Values",
             color='blue')
    plt.plot(subset_data.index[window_size:window_size + len(predictions)], predictions, label="Predicted Values",
             color='red', linestyle='--')
    plt.title("Actual vs. Predicted Values")
    plt.xlabel("Date")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.show()
    adaptive_mae = mean_absolute_error(actual_values, predictions)
    adaptive_mse = mean_squared_error(actual_values, predictions)
    print(adaptive_mse)
    print(adaptive_mae)

    return actual_values, predictions


def getPredictions(subset_data):
    actual_values, predictions = executeEvaluation(subset_data)
    return actual_values, predictions
