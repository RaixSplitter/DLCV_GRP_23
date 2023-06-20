import numpy as np
import sys
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import os
import datetime
from sklearn.metrics import f1_score
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from waste_dataset import WasteDatasetPatches
from simpleCNN import SimpleClassifier
from resnet18 import resnet50

patch_size = (128,128)
data = WasteDatasetPatches(resolution=patch_size)
train_size = int(0.60*len(data))
test_size = int(0.20*len(data))
val_size = len(data) - train_size - test_size

train_ds, test_ds, val_ds = random_split(data, [train_size, test_size, val_size])
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=10, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=10, shuffle=False)

toImage = transforms.ToPILImage()

for idx, (data, target) in enumerate(train_ds):
    if idx > 20: break
    
    img = toImage(data)
    img.save(os.path.join('out', f'img{idx}_label{target}.jpg'))