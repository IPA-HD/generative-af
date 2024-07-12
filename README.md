# generative-af
This repository contains code associated with [Generative Assignment Flows for Representing and Learning Joint Distributions of Discrete Data](https://arxiv.org/abs/2406.04527v1).
It was developed by the [Image \& Pattern Analysis Group](https://ipa.math.uni-heidelberg.de) at Heidelberg University.

## Configuration
Training hyperparameters need to be provided as JSON configuration files. Examples in `config/` serve as illustration for the required schema. To train a new model, define a configuration and run
```Bash
python train.py /path/to/training_config.json
```
Training artifacts, including model checkpoints, Tensorboard logs and hyperparameters are saved in `lightning_logs/`.

## Cityscapes Segmentation Data
To train a generative model for Cityscapes Segmentations, first download the [dataset](https://www.cityscapes-dataset.com/) to a directory of your choice and subsequently run the preprocessing routine
```Bash
cd data/image
python scale_cityscapes.py /path/to/raw/data 0.125 train
```
The second argument is a scaling factor for spatial dimensions (preprocessed files will be subsampled by a factor of 8 with interpolation mode `PIL.Image.NEAREST`). The number of segments will also be reduced, corresponding to the `category` of segments in the [Cityscapes torchvision dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Cityscapes.html).
Preprocessed Cityscapes segmentation data will be saved to `data/image/cityscapes/cityscapes_{split}_{scale}.pt`.

