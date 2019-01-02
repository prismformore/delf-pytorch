# delf-pytorch

This is a Pytorch implementation of DELF that is based on [official tensorflow implementation](https://github.com/tensorflow/models/tree/master/research/delf). This repository provides clean training codes on The [Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset).

According to the paper, training process is divided into two steps: 1. Train resnet backbone; 2. Train attention layers. You may choose this in settings.py

Dataset should be generated from Simplified Drawing files (.ndjson) of [Quick Draw Dataset](https://github.com/googlecreativelab/quickdraw-dataset).
