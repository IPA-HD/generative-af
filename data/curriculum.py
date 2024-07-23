"""
Initialize objects conforming to the Curriculum protocol.
"""
from .mnist import BinarizedMNIST
from .cityscapes import CityscapesSegmentations
from .simple_distr import (
    SampleGenerator, 
    CoupledBinaryDistribution,
    PinwheelDistribution,
    GaussianMixtureDistribution,
    SingleSimplexDistribution,
    StarkSimplexDistribution
)
from omegaconf import DictConfig
from typing import Protocol
from torch.utils.data import DataLoader
import torch

class Curriculum(Protocol):
    """
    Training Curricula can be datasets or sample generators 
    for simple distributions
    """
    def tensor_format(self) -> tuple[int, ...]:
        ...

    def dataloader(self, split: str = "train", **kwargs) -> DataLoader | SampleGenerator:
        ...

    def hist_from_samples(self, labelings: torch.Tensor) -> torch.Tensor:
        ...

def new_curriculum(dataset_params: DictConfig) -> Curriculum:
    """
    Returns a training curriculum based on the specified configuration.
    The instantiation of these objects does not load significant amounts of data from disk, 
    so it can be used to merely access dataset- or distribution metadata.
    """
    match dataset_params["dataset"]:
        case "mnist":
            return BinarizedMNIST(dataset_params)
        case "cityscapes":
            return CityscapesSegmentations(dataset_params)
        case "coupled_binary":
            return CoupledBinaryDistribution(dataset_params)
        case "pinwheel":
            return PinwheelDistribution(dataset_params)
        case "gaussian_mixture":
            return GaussianMixtureDistribution(dataset_params)
        case "simplex_toy":
            return SingleSimplexDistribution(dataset_params)
        case "simplex_stark":
            return StarkSimplexDistribution(dataset_params)
        case _:
            print("Unknown dataset", dataset_params["dataset"])
            raise NotImplementedError

