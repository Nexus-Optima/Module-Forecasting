import pandas as pd


def process_data(data):
    data = data
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data.set_index('Date', inplace=True)
    data = data.resample('D').mean().fillna(method='ffill')
    return data


def process_data_lagged(data):
    data = process_data(data)
    lags = [1, 7, 30]
    for col in data.columns:
        for lag in lags:
            data[f"{col}_lag{lag}"] = data[col].shift(lag)
    data.dropna(inplace=True)
    return data


def process_data_lagged_rolling_stats(data):
    data = process_data_lagged(data)
    window_sizes = [3, 14, 30]

    for window in window_sizes:
        data[f'Output_{window}d_mean'] = data['Output'].rolling(window=window).mean()
        data[f'Output_{window}d_std'] = data['Output'].rolling(window=window).std()

    data.dropna(inplace=True)
    return data
