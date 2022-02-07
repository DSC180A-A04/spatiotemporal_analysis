from torch import optim

from src.data import make_datasets
from src.models.model import QuantileLSTM

x, y = make_datasets()

input_size = 1
output_size = 1
hidden_size = 16
quantile = [0.025, 0.5, 0.975]
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
# model.fit(x, y, max_epochs=max_epochs, batch_size=batch_size)
# inference
y_cal_preds = model.predict(x_cal)
