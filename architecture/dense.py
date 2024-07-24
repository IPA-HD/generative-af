"""
Network architectures
"""
import torch.nn as nn
import torch
from .base import Vectorfield

class DenseNet(Vectorfield):
    """Dense Layers, ReLU activation, BatchNorm"""
    def __init__(self, c: int, n: int, hidden_dims: list[int], bias: bool = True):
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
