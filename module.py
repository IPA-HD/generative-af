import lightning as L
import json
import os
import torch
import torch.optim as optim
from torch.nn.functional import softmax
from torchvision.utils import make_grid # type: ignore
from af import integrate_af
from architecture import UNet, DenseNet, DirichletCNN, Vectorfield
from data.curriculum import new_curriculum
from util import labeling_as_colors, mean_free, mean_entropy, read
from flow_matching import matching_step
import matplotlib.pyplot as plt # type: ignore
from torch.distributions.gamma import Gamma
from omegaconf import DictConfig
from typing import Any, Sequence, Optional, cast

class FlowMatchingModule(L.LightningModule):
    """Flow matching base module"""
    def __init__(self, hparams: DictConfig):
        super(FlowMatchingModule, self).__init__()
        self.save_hyperparameters()

        # forward declarations for static analysis
        self.net: Vectorfield
        self.channel_dim: int

        self.model_params = hparams.model.params
        self.training_params = hparams.training
        self.integration_params = {
            "method": read("integrator", self.training_params, default="dopri5")
        }
        if self.integration_params["method"] in ["dopri5", "dopri8", "adaptive_heun", "bosh3", "fehlberg2"]:
            self.integration_params["atol"] = read("atol", self.training_params, default=1e-2)
            self.integration_params["rtol"] = read("rtol", self.training_params, default=1e-2)
        else:
            self.integration_params["options"] = {"step_size": read("step_size", self.training_params, default=1e-1)}
        self.t_end = read("t_end", self.training_params, default=1.0)
        # extract data format
        self.data = new_curriculum(hparams.data)
        self.tensor_format = self.data.tensor_format()
        self.data_dims = self.tensor_format[1:]

    def on_fit_start(self) -> None:
        self.log_time_density()

    def log_time_density(self) -> None:
        concentration = read("gamma_conc", self.training_params, default=2.0)
        rate = read("gamma_rate", self.training_params, default=0.4)
        g = Gamma(concentration, rate)  # type: ignore[no-untyped-call]
        t_end = self.training_params.t_end
        t = torch.linspace(0.0, t_end, steps=100)
        density = g.log_prob(t).exp()  # type: ignore[no-untyped-call]
        fig, ax = plt.subplots(1,1)
        ax.plot(t, density)
        ax.set_title(f"concentration {concentration}, rate {rate}")
        ax.set_xlabel("t")
        ax.set_ylabel("sampling density")
        fig.tight_layout()
        self.logger.experiment.add_figure("t_sampling_density", fig, self.global_step)  # type: ignore[union-attr]

    @torch.no_grad()
    def generate_samples(self, num_samples: int = 1, return_meta: bool = False, return_prob: bool = False) -> torch.Tensor | tuple[torch.Tensor, int, float]:
        device = self.device
        v0 = torch.randn(num_samples, *self.data_dims, device=device)
        self.net.reset_forward_count()
        sample_prob = integrate_af(v0, self.net, t_end=self.t_end, dim=self.channel_dim, **self.integration_params).detach()
        samples: torch.Tensor
        if return_prob:
            samples = sample_prob
        else:
            samples = sample_prob.argmax(dim=self.channel_dim)
        if return_meta:
            return samples, self.net.get_forward_count(), mean_entropy(sample_prob, dim=self.channel_dim).item()
        return samples

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = matching_step(batch, self.net, self.training_params, dim=self.channel_dim)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        assert self.training_params.method in ["SGD", "Adam", "AdamW"]
        if self.training_params.method == "SGD":
            optimizer = optim.SGD(self.net.parameters(), **self.training_params.opt_params)
        elif self.training_params.method == "Adam":
            optimizer = optim.Adam(self.net.parameters(), **self.training_params.opt_params)
        else:
            optimizer = optim.AdamW(self.net.parameters(), **self.training_params.opt_params)

        assert self.training_params.lr_scheduler in ["const", "CosineAnnealing"]
        if self.training_params.lr_scheduler == "const":
            return optimizer

        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.training_params.epochs)
        return {"optimizer": optimizer, "lr_scheduler": schedule}

class ImageFlow(FlowMatchingModule):
    """
    Assignment Flow with UNet payoff function
    for modelling Image-like data
    """
    def __init__(self, hparams: DictConfig):
        super(ImageFlow, self).__init__(hparams)
        assert self.data_dims[1] % 16 == 0
        assert self.data_dims[2] % 16 == 0
        self.c = self.tensor_format[1]
        self.net: Vectorfield = UNet(image_size=32, in_channels=self.c, out_channels=self.c, **self.model_params)
        self.channel_dim = self.net.channel_dim
        assert self.channel_dim in [1,2]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        num_samples = 3
        samples_rounded, fevals, sample_entr = self.generate_samples(num_samples, return_meta=True)
        sample_img_rgb = labeling_as_colors(samples_rounded.cpu(), self.c)
        sample_img_grid = make_grid(sample_img_rgb)
        self.logger.experiment.add_image('samples', sample_img_grid, self.global_step) # type: ignore
        self.log_dict({'num_fevals': fevals, 'sample_entr': sample_entr})

class ToyDataFlow(FlowMatchingModule):
    """
    Assignment Flow with Dense network payoff function
    for modelling short-sequence toy data
    """
    def __init__(self, hparams: DictConfig):
        super(ToyDataFlow, self).__init__(hparams)
        assert hparams.model.name in ["cnn_dirichlet", "dense"]
        self.net: Vectorfield
        if hparams.model.name == "cnn_dirichlet":
            assert hparams.data.dataset == "simplex_stark"
            self.c = hparams.data.num_classes
            self.n = 4
            self.net = DirichletCNN(self.c)
        else:
            self.c = self.tensor_format[1]
            self.n = self.tensor_format[2]
            self.net = DenseNet(self.c, self.n, **self.model_params)
        self.channel_dim = self.net.channel_dim
        assert self.channel_dim in [1,2]
        self.val_hist_accumulated: Optional[torch.Tensor]

    def on_validation_epoch_start(self) -> None:
        self.num_val_batches = 0
        self.val_hist_accumulated = None

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        num_hist_samples = 16000
        samples, fevals, sample_entr = self.generate_samples(num_hist_samples, return_meta=True)
        hist = self.data.hist_from_samples(samples.cpu())
        if self.val_hist_accumulated is None:
            self.val_hist_accumulated = hist.clone()
        else:
            self.val_hist_accumulated += hist
        self.num_val_batches += 1
        self.log("sample_entr", sample_entr, on_epoch=True, prog_bar=True)
        self.log("num_fevals", fevals)
    
    def on_validation_epoch_end(self) -> None:
        hist = cast(torch.Tensor, self.val_hist_accumulated) / self.num_val_batches
        kl = self.data.kl_from_hist(hist).item()
        if self.n == 2:
            # save histogram as figure
            fig, ax = plt.subplots(1,1,figsize=(5,5))
            ax.imshow(hist, vmin=0.0, cmap="Blues")
            ax.axis("off")
            fig.tight_layout()
            self.logger.experiment.add_figure("pmf_hist", fig, self.global_step) # type: ignore
        self.log("kl", kl, on_epoch=True, prog_bar=True)
