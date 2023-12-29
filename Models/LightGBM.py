import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def execute_lgbm(data, forecast_days):
    # train_size = int(0.8 * len(data))
    # train, test = data[:train_size], data[train_size:]
    #
    # train_X = train.drop('Output', axis=1)
    # train_y = train['Output']
    # test_X = test.drop('Output', axis=1)
    # test_y = test['Output']
    #
    # model = lgb.LGBMRegressor()
    # model.fit(train_X, train_y)
    #
    # predictions = model.predict(test_X)
    #
    # rmse = np.sqrt(mean_squared_error(test_y, predictions))
    # mae = mean_absolute_error(test_y, predictions)
    # print(f"RMSE: {rmse}")
    # print(mae)
    #
    # last_known_date = data.index[-1]
    # future_dates = [last_known_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    # forecasted_data = pd.DataFrame(index=future_dates)
    #
    # for col in data.columns:
    #     if col != 'Output':
    #         forecasted_data[col] = data[col].iloc[-forecast_days:].values
    # forecasted_values = model.predict(forecasted_data)
    #
    # # Plotting results
    # plt.figure(figsize=(15, 6))
    # plt.plot(train.index, train['Output'], label='Train')
    # plt.plot(test.index, test['Output'], label='Test')
    # plt.plot(test.index, predictions, label='Predicted', color='red')
    # plt.plot(forecasted_data.index, forecasted_values, label='Forecast', color='green')
    # plt.legend(loc='best')
    # plt.title("LightGBM Time Series Forecasting")
    # plt.show()
    #
    # return predictions, forecasted_values
    return