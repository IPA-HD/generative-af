import torch
import torch.nn as nn
from typing import Optional
from abc import abstractmethod

class Vectorfield(nn.Module):
    """
    Interface definition and type hints for neural network
    architectures defining assignment flow vector fields.
    """
    def __init__(self) -> None:
        super(Vectorfield, self).__init__()
        self.forward_count: int = 0
        self.channel_dim: int = 1

    @abstractmethod
    def counted_forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor]) -> torch.Tensor:
        self.forward_count += 1
        return self.counted_forward(x, timesteps)

    def reset_forward_count(self) -> None:
        self.forward_count = 0

    # Method to get the count
    def get_forward_count(self) -> int:
        return self.forward_count
