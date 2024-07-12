import torch
from torch.nn.functional import one_hot

class Threshold(object):
    """Threshold grayscale image at 0.5 to yield a binary labeling."""
    def __init__(self, threshold=0.5):
        super(Threshold, self).__init__()
        self.threshold = threshold
    
    def __call__(self, x):
        return (x > self.threshold).long().squeeze(0)

class SmoothSimplexCorners(object):
    """
    Move hard labeling toward simplex center by an epsilon
    """
    def __init__(self, eps, dim=1):
        super(SmoothSimplexCorners, self).__init__()
        self.eps = eps
        assert dim in [1,2]
        self.dim = dim

    def __call__(self, w):
        num_classes = w.shape[self.dim-1]
        return (1-self.eps)*w + self.eps*(1.0/num_classes)

class LabelingToAssignment(object):
    """
    One-hot encode labeling to hard assignment vector with channel
    dimension at first position.
    """
    def __init__(self, num_classes, dim=1):
        super(LabelingToAssignment, self).__init__()
        self.num_classes = num_classes
        assert dim in [1,2]
        self.dim = dim
    
    def __call__(self, labeling):
        W = one_hot(labeling, self.num_classes)
        if self.dim == 2:
            return W
        return torch.movedim(W, -1, 0)

