"""
Initialize objects conforming to the Curriculum protocol.
"""
from .mnist import BinarizedMNIST
from .cityscapes import CityscapesSegmentations
from .simple_distr import *

def new_curriculum(dataset_params):
	"""
	Returns a training curriculum based on the specified configuration.
	The instantiation of these objects does not load significant amounts of data from disk, 
	so it can be used to merely access dataset- or distribution metadata.
	"""
	if dataset_params["dataset"] == "mnist":
		return BinarizedMNIST(dataset_params)
	elif dataset_params["dataset"] == "cityscapes":
		return CityscapesSegmentations(dataset_params)
	elif dataset_params["dataset"] == "coupled_binary":
		return CoupledBinaryDistribution(dataset_params)
	elif dataset_params["dataset"] == "pinwheel":
		return PinwheelDistribution(dataset_params)
	elif dataset_params["dataset"] == "gaussian_mixture":
		return GaussianMixtureDistribution(dataset_params)
	elif dataset_params["dataset"] == "simplex_toy":
		return SingleSimplexDistribution(dataset_params)
	elif dataset_params["dataset"] == "simplex_stark":
		return StarkSimplexDistribution(dataset_params)
	else:
		print("Unknown dataset", dataset_params["dataset"])
		raise NotImplementedError

