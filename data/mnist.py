"""
MNIST data distribution.
"""
import torch
from torchvision.datasets import MNIST # type: ignore
import torchvision.transforms as tv_transforms # type: ignore
import data.transform as trf
import torch.utils.data as torchdata
from torch.utils.data import DataLoader, Dataset
import os
from typing import Any, Iterator, Optional
from omegaconf import DictConfig

class ImageOnlyDataLoader(torchdata.DataLoader[torch.Tensor]):
    """
    Wrapper for torchvision DataLoader which removes the labels from each batch.
    """
    def __init__(self, dataset: MNIST, **kwargs):
        super(ImageOnlyDataLoader, self).__init__(dataset, **kwargs)
    
    def __iter__(self) -> Iterator[torch.Tensor]:  # type: ignore[override]
        base_iterator = super(ImageOnlyDataLoader, self).__iter__()
        return map(lambda x: x[0], base_iterator)

class BinarizedMNIST(object):
    """
    Binary segmentations of MNIST images produced through trf.thresholding
    """
    def __init__(self, dataset_params: DictConfig):
        super(BinarizedMNIST, self).__init__()
        self.dataset_params = dataset_params
        self.spatial_dims = (32, 32)
        self.dataset: Optional[MNIST] = None
        self.num_classes = 2
        self.smoothing = 0.01

    def load_data(self, split: str = "train"):
        self.transforms = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Pad(2),
            trf.Threshold(),
            trf.LabelingToAssignment(self.num_classes),
            trf.SmoothSimplexCorners(self.smoothing)
        ])
        assert split in ["train", "val", "test"]
        if split == "val":
            print("Warning: selected \"val\" split for MNIST. This is an alias for the test set.")
        data_dir = "data/image/mnist"
        if not os.path.isdir(data_dir):
            os.makedirs("data/image/mnist")
        self.dataset = MNIST(root=data_dir, train=(split == "train"), transform=self.transforms, download=True)

        if self.dataset_params["restrict_digit"] >= 0:
            # restrict to single digit
            assert self.dataset_params["restrict_digit"] in list(range(10))
            idx = self.dataset.targets == self.dataset_params["restrict_digit"]
            self.dataset.data = self.dataset.data[idx]
            self.dataset.targets = self.dataset.targets[idx]

    def __len__(self) -> int:
        if self.dataset is None:
            self.load_data()
        ds: MNIST = self.dataset
        return len(ds)

    def tensor_format(self) -> tuple[int, int, int, int]:
        """
        Loaded data will have this tensor shape. 
        Batch dimension(s) are indicated by -1.
        """
        return (-1, 2, *self.spatial_dims)

    def dataloader(self, split: str = "train", **kwargs):
        if self.dataset is None:
            self.load_data(split)
        return ImageOnlyDataLoader(self.dataset, shuffle=(split == "train"), **kwargs)

    def hist_from_samples(self, labelings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def kl_from_hist(self, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

