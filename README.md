# generative-af
This repository contains code associated with [Generative Assignment Flows for Representing and Learning Joint Distributions of Discrete Data](https://arxiv.org/abs/2406.04527v1).
It was developed by the [Image \& Pattern Analysis Group](https://ipa.math.uni-heidelberg.de) at Heidelberg University.

## Configuration
Training hyperparameters are specified by YAML configuration files in `config/`. We use [hydra](https://hydra.cc/docs/intro/) to parse these files hierarchically, which also allows overwriting from the command line.

### Training Examples
*Binarized MNIST*
```Bash
python train.py data=mnist logging=epochs model=unet training=mnist
```
*Cityscapes Segmentations*
```Bash
python train.py data=cityscapes logging=steps model=unet training=cityscapes
```
*Simple Data Distributions*
```Bash
python train.py data=simple data.dataset=pinwheel logging=frequent model=dense training=simple
```
Options for `data.dataset` are `pinwheel`, `gaussian_mixture` and `coupled_binary`.

Training artifacts, including model checkpoints, Tensorboard logs and hyperparameters are saved in `lightning_logs/`.

## Cityscapes Segmentation Data
To train a generative model for Cityscapes Segmentations, first download the [dataset](https://www.cityscapes-dataset.com/) to a directory of your choice and subsequently run the preprocessing routine
```Bash
cd data/image
python scale_cityscapes.py /path/to/raw/data 0.125 train
```
The second argument is a scaling factor for spatial dimensions (preprocessed files will be subsampled by a factor of 8 with interpolation mode `PIL.Image.NEAREST`). The number of segments will also be reduced, corresponding to the `category` of segments in the [Cityscapes torchvision dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.Cityscapes.html).
Preprocessed Cityscapes segmentation data are saved to `data/image/cityscapes/cityscapes_{split}_{scale}.pt`.

