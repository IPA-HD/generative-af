"""
Likelihood under flow-matched assignment flows.
"""
import torch
import torch.nn as nn
import numpy as np
import math
from util import mean_free, standard_normal_logprob
from cnf import ODEfunc, CNF
from torch.nn.functional import softmax, log_softmax
from scipy.stats import chi2  # type: ignore[import-untyped]
from architecture import Vectorfield

def orthonormal_tangent_basis(c: int):
    B = torch.vstack((torch.eye(c-1), -torch.ones(1, c-1)))
    Q, _ = torch.linalg.qr(B)
    return Q

def coord_to_tangent(Q: torch.Tensor, vm: torch.Tensor, dim: int = 1):
    """
    tangent vector represented by coordinates vm in the basis Q
    """
    assert dim in [1,2]
    if dim == 2:
        # sequence dimension at second position
        return torch.einsum("ij,bsj...->bsi...", Q, vm)
    return torch.einsum("ij,bj...->bi...", Q, vm)

def tangent_to_coord(Q: torch.Tensor, v: torch.Tensor, dim: int = 1):
    """
    coordinates of tangent vector v in the basis Q
    """
    assert dim in [1,2]
    if dim == 2:
        # sequence dimension at second position
        return torch.einsum("ji,bsj...->bsi...", Q, mean_free(v))
    return torch.einsum("ji,bj...->bi...", Q, mean_free(v))

def af_as_cnf(net: Vectorfield, Q: torch.Tensor, t_end: float = 1.0, 
        div_method: str = "hutchinson_gauss", hutchinson_samples: int = 100, 
        atol: float = 1e-2, rtol: float = 1e-2, 
        dim: int = 1, reverse_time: bool = True):

    c = Q.shape[0]

    def diffeq(t: torch.Tensor, y: torch.Tensor):
        v = coord_to_tangent(Q, y, dim=dim)
        w = softmax(v, dim=dim)
        ts = t.expand(y.shape[0])
        F = mean_free(net(w, timesteps=ts), dim=dim)
        return tangent_to_coord(Q, F, dim=dim)

    odefunc = ODEfunc(diffeq, div_method=div_method, hutchinson_samples=hutchinson_samples)
    cnf = CNF(odefunc, T=t_end, rtol=rtol, atol=atol)
    return cnf

def radius_decision_disk(v_alpha: torch.Tensor, dim: int = 1):
    """
    Largest radius r of a ball around v_alpha (1,c,...) in the tangent space
    which does not leave the decision cone for the current label.
    """
    assert dim in [1,2]
    c = v_alpha.shape[dim]
    r = torch.linalg.vector_norm(v_alpha, dim=dim).flatten()[0]
    return r*math.sqrt(c/(2*(c-1))) # on single simplex

def sigma_from_disk_mass(dim: int, r: float , mass: float):
    b = chi2.ppf(mass, dim)
    return r / math.sqrt(b)

def sample_log_likelihood(
    net: Vectorfield, w_alpha: torch.Tensor, num_samples: int, proposal_mass: float = 0.8,
    div_method: str = "exact_ek", hutchinson_samples: int = 100,
    dim: int = 1, reduce_samples: bool = True, rounding: bool = True,
    rtol: float = 1e-2, atol: float = 1e-2, t_end: float = 1.0):

    c = w_alpha.shape[dim]
    device = w_alpha.device
    assert w_alpha.shape[0] == 1
    assert dim in [1,2]
    Q = orthonormal_tangent_basis(c).to(w_alpha.device)
    n = np.prod(w_alpha.shape)/c
    domain_ndim = (c-1)*n

    # draw proposal samples
    v_alpha = mean_free(torch.log(w_alpha), dim=dim)
    m_alpha = tangent_to_coord(Q, v_alpha, dim=dim)
    r = radius_decision_disk(v_alpha, dim=dim)
    simplex_mass = math.exp(math.log(proposal_mass)/n)
    proposal_sigma = sigma_from_disk_mass(c-1, r, simplex_mass)
    if dim == 1:
        assert w_alpha.ndim >= 2
        vm1 = torch.randn(num_samples, c-1, *w_alpha.shape[2:], device=device)
    else:
        assert w_alpha.ndim == 3
        vm1 = torch.randn(num_samples, w_alpha.shape[1], c-1, device=device)
    log_rho_v = standard_normal_logprob(vm1).reshape(num_samples, -1).sum(1) - domain_ndim*math.log(proposal_sigma)
    vm1 = proposal_sigma*vm1 + m_alpha

    # continuous change of variables
    cnaf = af_as_cnf(net, Q, t_end=t_end, div_method=div_method, hutchinson_samples=hutchinson_samples, dim=dim, rtol=rtol, atol=atol)
    zero = torch.zeros(num_samples, 1, device=device)
    with torch.no_grad():
        z, delta_logp = cnaf(vm1, zero, reverse=True)

    wz = softmax(coord_to_tangent(Q, z, dim=dim), dim=dim)
    logpz = standard_normal_logprob(z).reshape(num_samples, -1).sum(dim=1)
    logp1 = logpz - delta_logp.squeeze(-1)

    v1 = coord_to_tangent(Q, vm1, dim=dim)
    if rounding:
        # indicator function of label cone
        # this only contains 0 and 1 entries
        indicator = (v1.argmax(dim=dim) == w_alpha.argmax(dim=dim)).reshape(num_samples, -1).all(dim=1).float()
        log_lh_samples = torch.log(indicator) + logp1 - log_rho_v
    else:
        logW = log_softmax(v1, dim=dim)
        alpha = w_alpha.argmax(dim=dim, keepdim=True)
        alpha = alpha.expand(num_samples, *([-1]*(alpha.ndim-1)))
        log_lh_samples = logp1 - log_rho_v + torch.gather(logW, dim, alpha).reshape(num_samples, -1).sum(dim=1)

    if not reduce_samples:
        # return individual likelihood samples for aggregation
        return log_lh_samples

    log_lh = torch.logsumexp(log_lh_samples, dim=0) - math.log(num_samples)
    return log_lh

if __name__ == '__main__':
    import argparse
    import os.path
    from module import ImageFlow
    from tqdm import tqdm
    from util import loglh_to_bitsperdim, generate_random_id
    from data.curriculum import new_curriculum
    from data.mnist import BinarizedMNIST

    parser = argparse.ArgumentParser(
        prog='ComputeLikelihood',
        description='Compute likelihoods of binarized MNIST testset.')
    parser.add_argument("checkpoint", type=str, help="Filepath for the trained model.")
    parser.add_argument('--no_rounding', action='store_true', help="Compute likelihood for non-rounding model.")
    parser.add_argument('--samples', type=int, default=100, help="Batch size for sampling.")
    parser.add_argument('--hutchinson_samples', type=int, default=1, help="Number of samples for trace estimation.")
    parser.add_argument('--limit_data', type=int, default=None, help="Only use this many test data.")
    parser.add_argument('--method', type=str, 
        default="hutchinson_gauss",
        choices=["hutchinson_gauss", "hutchinson_rademacher", "exact_ek", "exact_jac"],
        help="Number of sampling rounds."
    )
    parser.add_argument('--proposal_mass', type=float, default=0.5, help="Mass of disk in decision cone under proposal distribution.")
    parser.add_argument('--rtol', type=float, default=1e-2, help="rtol for adjoint integration.")
    parser.add_argument('--atol', type=float, default=1e-2, help="atol for adjoint integration.")


    args = parser.parse_args()

    model_dir = os.path.dirname(args.checkpoint)
    parent_dir = os.path.dirname(model_dir)
    s = ImageFlow.load_from_checkpoint(args.checkpoint)
    hparams = s.hparams["hparams"]
    device = s.device

    assert hparams["data"]["dataset"] == "mnist"
    ds = BinarizedMNIST(hparams["data"])
    dl = ds.dataloader(split="test", batch_size=1)

    # how many test data are evaluated
    if limit := args.limit_data:
        total_data = min(limit, len(ds))
    else:
        total_data = len(ds)

    bpd_results = np.zeros(total_data)
    for test_ind, w in tqdm(enumerate(dl), total=total_data):
        llh = sample_log_likelihood(
            s.net, w.to(device), 
            dim=s.channel_dim,
            num_samples=args.samples,
            proposal_mass=args.proposal_mass,
            div_method=args.method,
            hutchinson_samples=args.hutchinson_samples,
            reduce_samples=True,
            atol=args.atol,
            rtol=args.rtol,
            rounding=(not args.no_rounding),
            t_end=hparams["training"]["t_end"]
        ).item()
        bpd = loglh_to_bitsperdim(llh, 32*32)
        print(test_ind, "log-likelihood", llh, "=", bpd, "bits / dim")
        bpd_results[test_ind] = bpd

        if test_ind == total_data - 1:
            break

    save_id = generate_random_id()
    fname = f"bpd_{args.method}_{total_data}test_{save_id}.npz"
    #fpath = os.path.join(args.model, fname)
    fpath = os.path.join(parent_dir, fname)
    np.savez(fpath,
        bpd=bpd_results,
        samples=args.samples,
        hut_samples=args.hutchinson_samples,
        atol=args.atol,
        rtol=args.rtol,
        proposal_mass=args.proposal_mass,
        rounding=(0.0 if args.no_rounding else 1.0)
    )
