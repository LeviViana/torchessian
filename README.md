# Torchessian

This repository aims to provide a tool for analyzing the full Hessian of the loss function for neural networks. For the moment, only a single GPU-mode is enabled, but I have plans to implement a distributed version in the future.

# Motivation

I found [this article]([https://arxiv.org/pdf/1901.10159.pdf](https://arxiv.org/pdf/1901.10159.pdf)) very interesting, and I wanted to reproduce the results very quickly. I've implemented a *batch mode* spectrum estimation in order to get even faster results. 