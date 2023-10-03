import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def min_max_scaler(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val), min_val, max_val


def inverse_min_max_scaler(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def execute_lstm(data):
    data = data.reset_index()
    data.drop(columns=['Date', 'Year', 'Month', 'Day', 'Season'], inplace=True)
    scaled_data, data_min, data_max = min_max_scaler(data.values)
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)
    look_back = 5

    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    model = LSTMModel(input_size=X_train.shape[2], hidden_size=90, num_layers=3, output_size=1)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.unsqueeze(1))
        loss.backward()
        optimizer.step()

    model.eval()
    train_predictions = model(X_train_tensor).detach().numpy()
    test_predictions = model(X_test_tensor).detach().numpy()

    train_predictions_orig = inverse_min_max_scaler(train_predictions, data_min[0], data_max[0])
    y_train_orig = inverse_min_max_scaler(y_train, data_min[0], data_max[0])
    test_predictions_orig = inverse_min_max_scaler(test_predictions, data_min[0], data_max[0])
    y_test_orig = inverse_min_max_scaler(y_test, data_min[0], data_max[0])

    train_mse = np.mean((train_predictions_orig - y_train_orig) ** 2)
    test_mse = np.mean((test_predictions_orig - y_test_orig) ** 2)

    forecast = []
    last_data = scaled_data[-look_back:]
    for _ in range(45):
        with torch.no_grad():
            model.eval()
            prediction = model(torch.FloatTensor(last_data[-look_back:].reshape(1, look_back, -2)))
            forecast.append(prediction.item())
            prediction_reshaped = prediction.numpy().flatten()
            new_row = np.hstack([last_data[-1, 1:], prediction_reshaped])
            last_data = np.vstack([last_data, new_row])
    forecast_orig = inverse_min_max_scaler(np.array(forecast).reshape(-1, 1), data_min[0], data_max[0])
    print(forecast_orig)

    all_actual = np.concatenate([y_train_orig, y_test_orig])
    all_predictions = np.concatenate([train_predictions_orig, test_predictions_orig])
    time_array = np.arange(len(all_actual) + len(forecast_orig))
    plt.figure(figsize=(16, 7))
    plt.plot(time_array[:len(all_actual)], all_actual, label='Actual Values', color='blue')
    plt.plot(time_array[:len(all_predictions)], all_predictions, label='Predicted Values', color='red', linestyle='--')
    plt.plot(time_array[-len(forecast_orig):], forecast_orig, label='Forecasted Values', color='green', linestyle='-.')
    plt.title('Actual vs. Predicted vs. Forecasted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Output Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return forecast_orig
