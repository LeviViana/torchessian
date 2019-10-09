# Torchessian

This repository aims to provide a tool for analyzing the full Hessian of the loss function for neural networks. For the moment, only a single GPU-mode is enabled, but I have plans to implement a distributed version in the future.

# Motivation

I found [this article](https://arxiv.org/pdf/1901.10159.pdf) very interesting, and I wanted to reproduce the results very quickly. I've implemented a *batch mode* spectrum estimation in order to get even faster results. 

For instance, when analyzing the impact of having batchnom layers in a ResNet-18 architecture, I found the following spectrum on the test set of CIFAR-10:

**Note**: both architectures (i.e. with and without batchnorm) were trained until reaching a global optimum, i.e. at least 98% of accuracy.

## Spectrum of a single batch 
![alt text](https://wintics-email-logo.s3.eu-west-3.amazonaws.com/batch_mode.png)

## Spectrum of the entire test dataset
![alt text](https://wintics-email-logo.s3.eu-west-3.amazonaws.com/complete_mode.png)

The results are pretty similar, and the conclusion is the same: the batchnorm layers seem to eliminate big positive eigenvalues, which makes the training process easier. 
