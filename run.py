import torch

from src.data import make_datasets, normalize
from src.models.inference import conformal_prediction
from src.models.train_model import train_model
from src.visualization import plot

datasets = make_datasets()
x_train, x_cal, x_test, y_train, y_cal, y_test = normalize(*datasets)

# train model
quantiles = torch.Tensor([0.025, 0.5, 0.975])
model = train_model(x_train, y_train, quantiles)

# inference
conformal_intervals = conformal_prediction(model,
                                           x_cal,
                                           y_cal,
                                           significance=0.1)

# visualize prediction results
plot(x_cal, y_cal, conformal_intervals, quantiles, save_plot=True)
