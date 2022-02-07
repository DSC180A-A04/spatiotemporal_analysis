import torch
import numpy as np


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
