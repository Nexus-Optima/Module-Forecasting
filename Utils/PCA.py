import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

data_multivar = pd.read_csv("../Data/ICAC multiple variables.csv", parse_dates=['Date'], dayfirst=True)
data_multivar.set_index('Date', inplace=True)
data_multivar.sort_index(inplace=True)

lags = [1, 2, 3, 7, 14, 21, 28]
for lag in lags:
    data_multivar[f"lag_{lag}"] = data_multivar['Cotlook_A_index'].shift(lag)

window_sizes = [7, 14, 21, 28]
for window in window_sizes:
    data_multivar[f"rolling_mean_{window}"] = data_multivar['Cotlook_A_index'].rolling(window=window).mean()
    data_multivar[f"rolling_std_{window}"] = data_multivar['Cotlook_A_index'].rolling(window=window).std()

data_multivar_fe = data_multivar.dropna()

features = data_multivar_fe.drop(columns=["Cotlook_A_index"])
pca = PCA(n_components=10)
pca.fit(features)

train_size = int(0.8 * len(data_multivar_fe))
train, test = data_multivar_fe.iloc[:train_size], data_multivar_fe.iloc[train_size:]

X_train, X_test = train.drop("Cotlook_A_index", axis=1), test.drop("Cotlook_A_index", axis=1)
y_train, y_test = train["Cotlook_A_index"], test["Cotlook_A_index"]
loadings = pca.components_.T
loading_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X_train.columns)
print(loading_df)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(num_components_95)

pc1_loadings = pca.components_[0]
pc1_values = features.dot(pc1_loadings).sort_index()
print(pc1_values)

pc2_loadings = pca.components_[1]
pc2_values = features.dot(pc2_loadings).sort_index()
print(pc2_values)

pc3_loadings = pca.components_[2]
pc3_values = features.dot(pc3_loadings).sort_index()
print(pc3_values)

