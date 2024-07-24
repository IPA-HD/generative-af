"""
Assignment flow integration routines.
"""
import torch
from torch.nn.functional import softmax
from util import mean_free
from torchdiffeq import odeint # type: ignore
from architecture import Vectorfield

def integrate_af(v0: torch.Tensor, F: Vectorfield, dim: int = 1, t_end: float = 1.0, **integrator_args):
    v = mean_free(v0, dim=dim)
    t = torch.zeros(v0.shape[0], device=v0.device)
    t1 = torch.ones(v0.shape[0], device=v0.device)
    field = lambda t, y: mean_free(F(softmax(y, dim=dim), timesteps=t*t1), dim=dim)
    v_end = odeint(field, v, torch.tensor([0.0, t_end], device=v0.device), **integrator_args)[-1,...]
    return softmax(v_end, dim=dim)
