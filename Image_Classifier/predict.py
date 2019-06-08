'''
Basic Usage (pwd: ImageClassifier): python predict.py flowers/test/1/image_06743.jpg checkpoint.pth 

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
parser = argparse.ArgumentParser(description='predict.py')
# Command line arguments
parser.add_argument('input', action='store', default = 'flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint', default='checkpoint.pth', action="store")
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

argsp = parser.parse_args()
input_image = argsp.input
num_out= argsp.top_k
power = argsp.gpu
path = argsp.checkpoint

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
dataloaders, dataset_sizes, image_datasets = setup_util.load_data()

model = setup_util.load_checkpoint(path)

probs, labels, classes = setup_util.predict(cat_to_name, input_image, model, num_out)

probs_formatted = [ '%.4f' % elem for elem in probs]
print_dict = dict(zip(classes, probs_formatted))

print ('Here are the probabilities and classes as predicted by the model. Cheers!')
print(print_dict)
