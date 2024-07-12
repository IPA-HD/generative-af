"""
Flow-matching on W.
"""

import torch
from torch.nn.functional import softmax
from util import mean_free, replicator, read
from torch.distributions.gamma import Gamma

def matching_step(w1, net, training_config, dim=1):
    device = w1.device
    batch_size = w1.shape[0]

    # inverse lift tangent vector of smoothed labeling
    # point on the e-geodesic connecting latent with 
    # smoothed labeling at time t
    v1 = mean_free(torch.log(w1), dim=dim)

    # draw t from Gamma distribution
    concentration = read("gamma_conc", training_config, default=2.0)
    rate = read("gamma_rate", training_config, default=0.4)
    g = Gamma(concentration, rate)
    t = g.sample((batch_size, *([1]*(w1.ndim-1)))).to(device)

    magnitude = read("field_magnitude", training_config, default=1.0)
    v1 /= torch.linalg.vector_norm(v1, dim=dim, keepdim=True)
    vt = magnitude*t*v1 + mean_free(torch.randn_like(v1), dim=dim)

    wt = softmax(vt, dim=dim)

    predicted_v = net(wt, t.flatten())
    vdiff = predicted_v-magnitude*v1
    loss = (vdiff*replicator(wt, vdiff, dim=dim)).reshape(batch_size, -1).sum(dim=1)

    # reduce loss as mean over batch dimension
    loss = loss.mean()

    return loss
