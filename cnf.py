"""
This code is modified from
https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py
and
https://github.com/rtqichen/ffjord/blob/master/lib/layers/cnf.py
which is under the following license.

MIT License

Copyright (c) 2018 Ricky Tian Qi Chen and Will Grathwohl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from icecream import ic
from torch.autograd.functional import jacobian
from tqdm import tqdm

def divergence_bf_jac(fx, y, **unused_kwargs):
    div = torch.zeros(y.shape[0], 1)
    for i in tqdm(range(y.shape[0])):
        # sequential over batch dimension
        J = jacobian(fx, y[i:(i+1),...])
        div[i,0] = torch.trace(J.reshape(2,2))
    return div

def divergence_bf_unit(f, y, **unused_kwargs):
    """
    Brute-force exact divergence by backward passes of unit
    vectors. This is parallel over the batch dimension and
    sequential over the spatial+label dimensions
    """
    batch_size = y.shape[0]
    div = torch.zeros(batch_size, 1, device=y.device)
    nc = np.prod([d for d in y.shape[1:]])
    ek = torch.zeros(batch_size, nc, device=y.device)
    for k in range(nc): #tqdm(range(nc)):
        if k > 0:
            ek[:,k-1] = 0.0
        ek[:,k] = 1.0
        func_output = f(y)
        func_output.backward(ek.reshape(y.shape), retain_graph=True)
        div[:,0] += y.grad.reshape(batch_size,-1)[:,k]
        y.grad.zero_()
    #print("divergence", div.flatten())
    return div

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, y, e=None):
    # hutchinson estimator with multiple draws
    approx_div = torch.zeros(y.shape[0], 1, device=y.device)
    num_samples = y.shape[0]
    y = y.detach().requires_grad_(True)
    for draw_i in range(e.shape[0]): #tqdm(range(e.shape[0])):
        z = e[draw_i,...]
        func_output = f(y)
        func_output.backward(z, retain_graph=True)
        approx_div += (y.grad * z).reshape(num_samples, -1).sum(dim=1, keepdim=True)
        y.grad.zero_()  # Reset gradients for next iteration
    approx_div /= e.shape[0]
    #print("approx divergence", approx_div.flatten())
    return approx_div

# def divergence_approx(f, y, e=None):
#     e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
#     e_dzdx_e = e_dzdx * e
#     approx_tr_dzdx = e_dzdx_e.reshape(y.shape[0], -1).sum(dim=1)
#     return approx_tr_dzdx

def sample_rademacher(y, num_repetitions=100):
    return torch.sign(sample_gaussian(y, num_repetitions))

def sample_gaussian(y, num_repetitions=100):
    return torch.randn(num_repetitions, *y.shape, device=y.device)

class ODEfunc(nn.Module):

    def __init__(self, diffeq, div_method="hutchinson_gauss", hutchinson_samples=100, residual=False, rademacher=False):
        super(ODEfunc, self).__init__()
        
        # self.diffeq = diffeq_layers.wrappers.diffeq_wrapper(diffeq)
        self.diffeq = diffeq
        self.residual = residual
        self.div_method = div_method
        self.hutchinson_samples = hutchinson_samples

        assert div_method in ["hutchinson_gauss", "hutchinson_rademacher", "exact_ek", "exact_jac"]

        self.register_buffer("_num_evals", torch.tensor(0.))

        self._e = None

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, states):
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = t.clone().detach().to(y)
        batchsize = y.shape[0]

        # Select divergence method
        if self.div_method == "hutchinson_gauss":
            self.divergence_fn = divergence_approx
            # Sample and fix the noise
            if self._e is None:
                self._e = torch.randn(self.hutchinson_samples, *y.shape, device=y.device)
        elif self.div_method == "hutchinson_rademacher":
            self.divergence_fn = divergence_approx
            # Sample and fix the noise
            if self._e is None:
                self._e = torch.sign(torch.randn(self.hutchinson_samples, *y.shape, device=y.device))
        elif self.div_method == "exact_ek":
            self.divergence_fn = divergence_bf_unit
        elif self.div_method == "exact_jac":
            self.divergence_fn = divergence_bf_jac
        else:
            raise NotImplementedError

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            for s_ in states[2:]:
                s_.requires_grad_(True)
            dy = self.diffeq(t, y, *states[2:])
            divergence = self.divergence_fn(lambda x: self.diffeq(t, x), y, e=self._e)
        if self.residual:
            dy = dy - y
            divergence -= torch.ones_like(divergence) * torch.tensor(np.prod(y.shape[1:]), dtype=torch.float32
                                                                     ).to(divergence)
        return tuple([dy, -divergence] + [torch.zeros_like(s_).requires_grad_(True) for s_ in states[2:]])

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()

def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
