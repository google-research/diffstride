# Diffstride

This is the code for the ICLR22 paper [Learning Strides in Convolutional Neural Network](https://openreview.net/forum?id=M752z9FKJP) by R.Riad, D. Grangier, O. Teboul and N. Zeghidour.

## Installation

To install the diffstride library, run the following `pip` git clone this repo:

```
git clone https://github.com/google-research/diffstride.git
```

The cd into the root and run the command:
```
pip install -e .
```

## Example training

To run an example training on CIFAR10 and save the result in TensorBoard:

```
python3 -m diffstride.example.main \
  --gin_config=cifar10.gin \
  --gin_bindings="train.workdir=/tmp/exp/diffstride/resnet18/"
```

## CPU/GPU Warning
We rely on the tensorflow FFT implementation which requires the input data to be in the `channels_first` format. This is usually not the regular data format of most datasets (including CIFAR) and running with `channels_first` also prevents from using of convolutions on CPU. Therefore even if we do support `channels_last` data format for CPU compatibility , we do encourage the user to run with `channels_first` data format *on GPU*.


## Disclainer
This is not an official Google product.

