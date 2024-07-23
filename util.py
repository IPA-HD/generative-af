import math
import torch
from torch.nn.functional import softmax
from matplotlib.cm import ScalarMappable # type: ignore
from torch.special import entr
import random
import string
from omegaconf import DictConfig
from typing import Any, Iterator

class DummyDataloader(object):
    """for validation without data"""
    def __init__(self, num_val_batches: int = 1):
        super(DummyDataloader, self).__init__()
        self.num_val_batches = num_val_batches

    def __iter__(self) -> Iterator[Any]:
        return iter(self.num_val_batches*[None])

    def __len__(self) -> int:
        return self.num_val_batches

def generate_random_id(size: int = 4) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(size))

def mean_entropy(w: torch.Tensor, dim: int = 1) -> torch.Tensor:
    c = w.shape[dim]
    entropy: torch.Tensor = entr(w).sum(dim=dim).mean() / math.log(c)
    return entropy

def logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    lme: torch.Tensor = torch.logsumexp(x, dim=dim) - torch.tensor([math.log(x.shape[dim])])
    return lme

def read(key: str, config: DictConfig, default: Any = None) -> Any:
    if key in config.keys():
        return config[key]
    print(f"WARNING: \"{key}\" not configured, defaulting to {default}")
    return default

def labeling_as_colors(x: torch.Tensor, c: int) -> list[torch.Tensor]:
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    assert x.ndim == 3 # batch dimensions is assumed
    sm = ScalarMappable(cmap='gray' if c == 2 else 'tab20b')
    sm.set_clim(0.0, c-1)
    rgb_vals = [sm.to_rgba(x[i,...]) for i in range(x.shape[0])]
    rgb_tensors = [torch.movedim(torch.from_numpy(y), -1, 0) for y in rgb_vals]
    return rgb_tensors

def loglh_to_bitsperdim(log_lh: torch.Tensor | float, dim: int) -> torch.Tensor | float:
    return -log_lh / math.log(2) / dim

def lift(W: torch.Tensor, V: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return softmax(V + torch.log(W), dim=dim)

def mean_free(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Project vector to the tangent space T_0W.
    """
    xp: torch.Tensor = x - x.mean(dim=dim, keepdim=True)
    return xp

def standard_normal_logprob(z: torch.Tensor) -> torch.Tensor:
    logZ: torch.Tensor = -0.5 * math.log(2 * math.pi) - z.pow(2) / 2
    return logZ 

def replicator(s0: torch.Tensor, v: torch.Tensor, dim: int = 1) -> torch.Tensor:
    assert v.ndim == s0.ndim
    s0v = s0*v
    Rs: torch.Tensor = s0v - s0v.sum(dim=dim, keepdim=True)*s0
    return Rs
