import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


def make_datasets():
    # generate linear time series data with some noise
    n = 200
    x_max = 10
    slope = 2
    scale = 2

    x = torch.from_numpy(
        np.linspace(-x_max, x_max, n).reshape(-1, 1).astype(np.float32))
    y = slope * x + np.random.normal(0, scale, n).reshape(-1, 1).astype(
        np.float32)

    # split to three sections
    window_size = n // 3
    if n % 3 == 0:
        # If an even split of data, then give each dataset n//3 entries
        x_train = x[:window_size]
        x_cal = x[window_size:window_size * 2]
        x_test = x[window_size * 2:]
        y_train = y[:window_size]
        y_cal = y[window_size:window_size * 2]
        y_test = y[window_size * 2:]
    elif n % 3 == 1:
        # if there's 1 left, it automatically gets added to the test set
        x_train = x[:window_size]
        x_cal = x[window_size:window_size * 2]
        x_test = x[window_size * 2:]
        y_train = y[:window_size]
        y_cal = y[window_size:window_size * 2]
        y_test = y[window_size * 2:]
    elif n % 3 == 2:
        # if there's 2 extra data points, add one to calibration, then other extra will go towards test
        x_train = x[:window_size]
        x_cal = x[window_size:window_size * 2 + 1]
        x_test = x[window_size * 2:]
        y_train = y[:window_size]
        y_cal = y[window_size:window_size * 2 + 1]
        y_test = y[window_size * 2:]

    return x_train, x_cal, x_test, y_train, y_cal, y_test


def normalize(x_train, x_cal, x_test, y_train, y_cal, y_test):
    scaler = StandardScaler()

    # Scale the y data locally (ex. train scaled to train)
    y_train_scaled = scaler.fit_transform(y_train)
    y_cal_scaled = scaler.fit_transform(y_cal)
    y_test_scaled = scaler.fit_transform(y_test)

    # Scale the x data locally (ex. train scaled to train)
    x_train_scaled = scaler.fit_transform(x_train)
    x_cal_scaled = scaler.fit_transform(x_cal)
    x_test_scaled = scaler.fit_transform(x_test)

    # Convert our scaled data into tensors of type float since that is what our torchTS model expects
    y_train = torch.tensor(y_train_scaled).float()
    y_cal = torch.tensor(y_cal_scaled).float()
    y_test = torch.tensor(y_test_scaled).float()

    x_train = torch.tensor(x_train_scaled).float()
    x_cal = torch.tensor(x_cal_scaled).float()
    x_test = torch.tensor(x_test_scaled).float()
    return x_train, x_cal, x_test, y_train, y_cal, y_test