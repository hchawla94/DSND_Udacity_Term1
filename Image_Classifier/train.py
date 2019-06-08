'''
Basic Usage (pwd: ImageClassifier): python train.py flowers

'''
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import PIL
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse
import os

import setup_util

# Initialize parser
parser = argparse.ArgumentParser(description='train.py')
# Command line arguments
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=15)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_unit_1', type=int, dest="hidden_unit_1", action="store", default=1500)
parser.add_argument('--hidden_unit_2', type=int, dest="hidden_unit_2", action="store", default=1000)
parser.add_argument('--hidden_unit_3', type=int, dest="hidden_unit_3", action="store", default=500)

argsp = parser.parse_args()

data_dir = argsp.data_dir
power = argsp.gpu
path = argsp.save_dir
lr = argsp.learning_rate
eps= argsp.epochs
network_name = argsp.arch
hidden_layer_1 = argsp.hidden_unit_1
hidden_layer_2 = argsp.hidden_unit_2
hidden_layer_3 = argsp.hidden_unit_3

dataloaders, dataset_sizes, image_datasets = setup_util.load_data()
model = setup_util.model_setup(network_name,0.6,hidden_layer_1, hidden_layer_2, hidden_layer_3, power)
criterion, optimizer, scheduler, eps = setup_util.model_params(model, lr, eps)

setup_util.train_model(model, dataloaders, dataset_sizes, image_datasets, criterion, optimizer, scheduler, eps, power)

setup_util.save_checkpoint(model, image_datasets, path, network_name, hidden_layer_1, hidden_layer_2, hidden_layer_3)

print("Model has been trained")