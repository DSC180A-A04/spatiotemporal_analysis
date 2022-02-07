import torch
from torch import nn
from torchts.nn.loss import quantile_loss
from torchts.nn.model import TimeSeriesModel


class QuantileLSTM(TimeSeriesModel):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 optimizer,
                 quantile=0.5,
                 **kwargs):
        super().__init__(optimizer,
                         criterion=quantile_loss,
                         criterion_args={"quantile": quantile},
                         **kwargs)
        n_quantiles = 1 if isinstance(quantile, float) else len(quantile)
        self.lstm = nn.ModuleList(
            [nn.LSTMCell(input_size, hidden_size) for _ in range(output_size)])
        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, n_quantiles) for _ in range(output_size)])

    def forward(self, x, y=None, batches_seen=None):
        hidden, _ = zip(*[m(x) for m in self.lstm])
        out = [m(h) for m, h in zip(self.linear, hidden)]
        return torch.hstack(out)
