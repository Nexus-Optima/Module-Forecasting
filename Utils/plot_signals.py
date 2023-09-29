import matplotlib.pyplot as plt
import pandas as pd
from Utils.generate_signals import generate_signals, modified_generate_signals


def plot_buy_sell_entire_data(data):
    data['diff'] = data['Output'].diff()
    window_size_diff = 7
    data['smoothed_diff'] = data['diff'].rolling(window=window_size_diff).mean()
    buy_threshold = data['smoothed_diff'].quantile(0.95)
    sell_threshold = data['smoothed_diff'].quantile(0.05)
    data['signal'] = 0
    data.loc[data['smoothed_diff'] > buy_threshold, 'signal'] = 1
    data.loc[data['smoothed_diff'] < sell_threshold, 'signal'] = -1
    plt.figure(figsize=(16, 8))
    plt.plot(data.index, data['Output'], label='Output', color='blue')
    plt.scatter(data[data['signal'] == 1].index, data[data['signal'] == 1]['Output'], label='Buy Signal', marker='^',
                color='g', alpha=1, s=100)
    plt.scatter(data[data['signal'] == -1].index, data[data['signal'] == -1]['Output'], label='Sell Signal', marker='v',
                color='r', alpha=1, s=100)

    plt.title('Buy and Sell signals on the entire dataset')
    plt.xlabel('Date')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_predicted_data(predictions, subset_data, predicted_output, future_data, window_size):
    predicted_signals = generate_signals(pd.Series(predictions))

    dates = pd.date_range(start=subset_data.index[window_size], periods=len(predictions), freq='D')

    forecasted_series = pd.Series(predicted_output, index=future_data.index)
    forecasted_signals = generate_signals(forecasted_series)

    predicted_series = pd.Series(predictions, index=dates)
    predicted_signals = pd.Series(predicted_signals)
    predicted_signals.index = predicted_series.index

    forecasted_signals = pd.Series(forecasted_signals)
    forecasted_signals.index = forecasted_series.index

    plt.figure(figsize=(14, 7))
    plt.plot(predicted_series, label="Predicted Values", color='red', linestyle='--')
    plt.plot(forecasted_series, label="Forecast", color='green', linestyle='-.')

    buy_signals = predicted_signals[predicted_signals == 1.0].index
    sell_signals = predicted_signals[predicted_signals == -1.0].index

    plt.scatter(buy_signals, predicted_series[buy_signals], label='BUY Signal', marker='^', color='g', alpha=1, s=100)
    plt.scatter(sell_signals, predicted_series[sell_signals], label='SELL Signal', marker='v', color='r', alpha=1,s=100)
    plt.title("Predicted and Forecasted Values with Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Output")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
