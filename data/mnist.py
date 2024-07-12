"""
MNIST data distribution.
"""
from torchvision.datasets import MNIST
import torchvision.transforms as tv_transforms
import data.transforms as trf
from torch.utils.data import DataLoader
import os

class ImageOnlyDataLoader(DataLoader):
    """Wrapper for torchvision DataLoader which removes the labels from each batch."""
    def __init__(self, dataset, **kwargs):
        super(ImageOnlyDataLoader, self).__init__(dataset, **kwargs)
    
    def __iter__(self):
        base_iterator = super(ImageOnlyDataLoader, self).__iter__()
        return map(lambda x: x[0], base_iterator)

class BinarizedMNIST(object):
    """
    Binary segmentations of MNIST images produced through trf.thresholding
    """
    def __init__(self, dataset_params):
        super(BinarizedMNIST, self).__init__()
        self.dataset_params = dataset_params
        self.spatial_dims = (32, 32)
        self.dataset = None
        self.num_classes = 2
        self.smoothing = dataset_params["integer_smoothing"]

    def load_data(self, split="train"):
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
            idx = mnist_dataset.targets == self.dataset_params["restrict_digit"]
            mnist_dataset.data = mnist_dataset.data[idx]
            mnist_dataset.targets = mnist_dataset.targets[idx]

    def tensor_format(self):
        """
        Loaded data will have this tensor shape. 
        Batch dimension(s) are indicated by -1.
        """
        return (-1, 2, *self.spatial_dims)

    def dataloader(self, split="train", **kwargs):
        if self.dataset is None:
            self.load_data(split)
        return ImageOnlyDataLoader(self.dataset, shuffle=(split == "train"), **kwargs)

