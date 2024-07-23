"""
Network architectures
"""
import torch.nn as nn
import torch
from .base import Vectorfield

class DenseNet(Vectorfield):
    """Dense Layers, ReLU activation, BatchNorm"""
    def __init__(self, c, n, hidden_dims, bias=True):
        super(DenseNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.c = c
        self.n = n
        self.channel_dim = 1
        feature_seq = [c*n+1, *hidden_dims, c*n]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(d_in) if len(hidden_dims) > 0 else nn.Identity(),
                nn.Linear(d_in, d_out, bias=bias),
                nn.ReLU() if len(hidden_dims) > 0 else nn.Identity()
            )
            for (d_in, d_out) in zip(feature_seq[:-1], feature_seq[1:])
        ])

    def accepts_data_format(self, batch_shape):
        if len(batch_shape) < 2:
            print("Tensor ndim missmatch")
            return False
        # dense models can technically accept any spatial and channel dimensions
        # but this is not the intended use, so at least issue a warning
        if (batch_shape[self.channel_dim] != self.c) or (batch_shape[2] != self.n):
            print("Warning: data used for training do not match expected spatial dimensions. Expected", (self.c, self.n), "got", batch_shape[1:])
        return True

    def counted_forward(self, x, timesteps=None):
        batch_size = x.shape[0]
        other_dims = x.shape[1:]
        if timesteps is None:
            x = torch.cat((x.reshape(batch_size, -1), torch.zeros(batch_size, 1, device=x.device)), dim=1)
        else:
            x = torch.cat((x.reshape(batch_size, -1), timesteps.unsqueeze(-1)), dim=1)
        for l in self.layers:
            x = l(x)
        return x.reshape(batch_size, *other_dims)
