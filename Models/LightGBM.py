import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def execute_lgbm(data):
    train_size = int(0.8 * len(data))
    train, test = data[:train_size], data[train_size:]

    train_X = train.drop('Output', axis=1)
    train_y = train['Output']
    test_X = test.drop('Output', axis=1)
    test_y = test['Output']

    model = lgb.LGBMRegressor()
    model.fit(train_X, train_y)

    predictions = model.predict(test_X)

    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    print(f"RMSE: {rmse}")
    print(mae)

    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train['Output'], label='Train')
    plt.plot(test.index, test['Output'], label='Test')
    plt.plot(test.index, predictions, label='Predicted', color='red')
    plt.legend(loc='best')
    plt.title("LightGBM Time Series Forecasting")
    plt.show()

    return predictions

