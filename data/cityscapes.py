import torch
import os.path
import data.transform as trf
import torchvision.transforms as tv_transforms # type: ignore
from torch.utils.data import TensorDataset, DataLoader

class TransformedTensorDataset(TensorDataset):
    """
    Wrapper for TensorDataset with transforms pipeline
    """
    def __init__(self, data_tensor, transforms=None):
        super(TransformedTensorDataset, self).__init__(data_tensor)
        self.transforms = transforms
    
    def __getitem__(self, idx):
        (x,) = super(TransformedTensorDataset, self).__getitem__(idx)
        if self.transforms is None:
            return x
        return self.transforms(x)

class CityscapesSegmentations(object):
    """
    Segmentations of cityscapes images, reduced to 8 class categories
    and with spatial downscaling
    """
    def __init__(self, dataset_params):
        super(CityscapesSegmentations, self).__init__()
        assert dataset_params["num_classes"] == 8
        self.num_classes = dataset_params["num_classes"]
        self.scale = dataset_params["scale"]
        self.spatial_dims = (int(1024*self.scale), int(2048*self.scale))
        self.tensor_data = None
        self.smoothing = dataset_params["integer_smoothing"]

    def load_data(self, split="train"):
        data_fpath = os.path.join(f"data/image/cityscapes/cityscapes_{split}_{self.scale}.pt")
        if not os.path.isfile(data_fpath):
            print(f"No cityscapes data at {data_fpath}")
            print("Please download cityscapes segmentations and preprocess them using data/image/scale_cityscapes.py")
        print("Loading cityscapes segmentations from", data_fpath)
        self.tensor_data = torch.load(data_fpath)
        self.transforms = tv_transforms.Compose([
            trf.LabelingToAssignment(self.num_classes),
            trf.SmoothSimplexCorners(self.smoothing)
        ])
        self.dataset = TransformedTensorDataset(self.tensor_data, self.transforms)

    def tensor_format(self):
        """
        Loaded data will have this tensor shape. 
        Batch dimension(s) are indicated by -1.
        """
        return (-1, self.num_classes, *self.spatial_dims)

    def dataloader(self, split="train", **kwargs):
        if self.tensor_data is None:
            self.load_data(split)
        return DataLoader(self.dataset, shuffle=(split == "train"), **kwargs)