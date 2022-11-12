# System
import os
import sys
import json
import importlib
from tqdm import tqdm

# Data
import numpy as np
import pandas as pd
from collections import Counter

# Visualization
import ipywidgets as widgets
import matplotlib.pyplot as plt

# Custom Modules
from scripts import crawler
from utils import file_utils, device_utils
from model import trainer, dataset, baseline_cnn_1, rnn_cnn, rnn_cnn_bi, rnn_cnn_2


import cv2
import torch
import torch.nn as nn
# from torchvision.models import ResNet18_Weights
from torchvision import transforms, models, datasets

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score

image_dir = 'data/images'

data = dataset.DataSet(max_num_img=300)
# data.max_num_img = None

data.load_all()

batch_size = 1
num_epoch = 1

rnn_model = rnn_cnn.CNNWithRNN(len(data.categories))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=5e-4)

mtrainer = trainer.Trainer(rnn_model, optimizer, criterion, data, batch_size)
mtrainer.run_train(num_epoch)

mtrainer.run_test()