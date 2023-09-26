# Time Series Decomposition
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('../../Data/ICAC multiple variables.csv')
data['Date'] = pd.to_datetime(data['Date'])
data_ts = data.set_index('Date')

data_ts_grouped = data_ts.groupby(data_ts.index).mean()
data_ts_filled = data_ts_grouped.resample('D').ffill()

result = seasonal_decompose(data_ts_filled['Output'], model='additive', period=365)
result.plot()
plt.tight_layout()
plt.show()

selected_features = ['IMPTS', 'S/MU', 'Year']
fig, axes = plt.subplots(nrows=len(selected_features), ncols=1, figsize=(14, 12))

for idx, feature in enumerate(selected_features):
    ax = axes[idx]
    ax2 = ax.twinx()
    ax.plot(result.trend, label='Trend', color='orange')
    ax.plot(result.seasonal, label='Seasonality', color='green')
    ax2.plot(data_ts_filled[feature], label=feature, color='blue', linestyle='--')
    ax.set_ylabel('Output Components')
    ax2.set_ylabel(feature)
    ax.set_title(f'{feature} vs. Trend and Seasonality Components')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

def lagged_correlation(series1, series2, max_lags):
    correlations = []
    for lag in range(0, max_lags+1):
        if lag == 0:
            # No lag
            corr = series1.corr(series2)
        else:
            # Compute correlation with lag
            corr = series1.corr(series2.shift(-lag))
        correlations.append(corr)
    return correlations


lag_correlations = {}
max_lags = 30
for feature in selected_features:
    feature_data = data_ts[feature]
    trend_corr = lagged_correlation(result.trend.dropna(), feature_data, max_lags)
    seasonal_corr = lagged_correlation(result.seasonal, feature_data, max_lags)
    lag_correlations[feature] = {'Trend': trend_corr, 'Seasonality': seasonal_corr}
fig, axes = plt.subplots(nrows=len(selected_features), ncols=1, figsize=(14, 12))

for idx, feature in enumerate(selected_features):
    ax = axes[idx]
    ax.plot(range(0, max_lags+1), lag_correlations[feature]['Trend'], label='Trend', marker='o')
    ax.plot(range(0, max_lags+1), lag_correlations[feature]['Seasonality'], label='Seasonality', marker='x')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(f'Lagged Correlation of {feature} with Trend and Seasonality Components')
    ax.set_xlabel('Lag (Days)')
    ax.set_ylabel('Correlation Coefficient')
    ax.legend()

plt.tight_layout()
plt.show()