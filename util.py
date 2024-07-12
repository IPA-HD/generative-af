import math
import torch
from torch.nn.functional import softmax
from matplotlib.cm import ScalarMappable
from torch.special import entr
import random
import string

class DummyDataloader(object):
    """for validation without data"""
    def __init__(self, num_val_batches=1):
        super(DummyDataloader, self).__init__()
        self.num_val_batches = num_val_batches

    def __iter__(self):
        return iter(self.num_val_batches*[None])

    def __len__(self):
        return self.num_val_batches

def generate_random_id(size=4):
    return "".join(random.choice(string.ascii_letters) for _ in range(size))

def mean_entropy(w, dim=1):
    c = w.shape[dim]
    return entr(w).sum(dim=dim).mean() / math.log(c)

def logmeanexp(x, dim):
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])

def read(key, config, default=None):
    if key in config.keys():
        return config[key]
    print(f"WARNING: \"{key}\" not configured, defaulting to {default}")
    return default

def labeling_as_colors(x, c):
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    assert x.ndim == 3 # batch dimensions is assumed
    sm = ScalarMappable(cmap='gray' if c == 2 else 'tab20b')
    sm.set_clim(0.0, c-1)
    rgb_vals = [sm.to_rgba(x[i,...]) for i in range(x.shape[0])]
    rgb_tensors = [torch.movedim(torch.from_numpy(y), -1, 0) for y in rgb_vals]
    return rgb_tensors

def loglh_to_bitsperdim(log_lh, dim):
    return -log_lh / math.log(2) / dim

def lift(W, V, dim=1):
    return softmax(V + torch.log(W), dim=dim)

def mean_free(x, dim=1):
    """
    Project vector to the tangent space T_0W.
    """
    return x - x.mean(dim=dim, keepdim=True)

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def replicator(s0, v, dim=1):
    assert v.ndim == s0.ndim
    s0v = s0*v
    return s0v - s0v.sum(dim=dim, keepdim=True)*s0
