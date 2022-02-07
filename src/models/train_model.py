import torch
from torch import optim
from src.models import QuantileLSTM


def train_model(x_train, y_train, quantiles=torch.Tensor([0.025, 0.5, 0.975])):
    input_size = 1
    output_size = 1
    hidden_size = 16
    optimizer = optim.Adam
    optimizer_args = {"lr": 0.005}
    max_epochs = 100
    batch_size = 10

    model = QuantileLSTM(
        input_size,
        output_size,
        hidden_size,
        optimizer,
        quantile=quantiles,
        optimizer_args=optimizer_args,
    )

    # train model
    model.fit(x_train, y_train, max_epochs=max_epochs, batch_size=batch_size)
    return model