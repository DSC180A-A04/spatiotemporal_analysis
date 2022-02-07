import torch
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.data import make_datasets
from src.models.model import QuantileLSTM

x_train, x_cal, x_test, y_train, y_cal, y_test = make_datasets()

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

input_size = 1
output_size = 1
hidden_size = 16
quantile = torch.Tensor([0.025, 0.5, 0.975])
optimizer = optim.Adam
optimizer_args = {"lr": 0.005}
max_epochs = 100
batch_size = 10

model = QuantileLSTM(
    input_size,
    output_size,
    hidden_size,
    optimizer,
    quantile=quantile,
    optimizer_args=optimizer_args,
)

# train model
model.fit(x_train, y_train, max_epochs=max_epochs, batch_size=batch_size)
# inference
y_cal_preds = model.predict(x_cal)

# visualize prediction results
n_quantiles = len(quantile)
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_index = [2, 1, 3]

plt.plot(x_cal, y_cal, label="y_true")

for i, c in zip(range(n_quantiles), color_index):
    plt.plot(x_cal,
             y_cal_preds[:, i],
             c=cycle_colors[c],
             label=f"p={quantile[i]}")

plt.legend()
plt.show()