import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
import sys
import os

import matplotlib.pyplot as plt
from IPython.display import clear_output

#Our imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_utils import DRIVEDataset, get_data_loader
from model.baseline import EncDec
from train import train, train_with_metrics
import ph2_dataloader

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'drive'
    current_path = os.getcwd()
    plot_path = os.path.join(current_path, '..', 'plots')
    results_path = os.path.join(current_path, '..', 'results')
    if dataset == 'drive':
        #Paths
        image_path = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'images')
        vessel_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', '1st_manual')
        eye_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'mask')

        size = 128 # Image size 
        train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = get_data_loader(image_path, vessel_mask, size)
    elif dataset == 'ph2':
        train_dataloader, test_dataloader, val_dataloader = ph2_dataloader.get_dataloaders()
    else:
        print('that\'s not a dataset!')
    learning_rate = 0.001
    epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncDec(image_size=size).to(device)
    train_with_metrics(model, optim.AdamW(model.parameters(), lr=learning_rate), bce_loss, epochs, train_dataloader, val_dataloader, results_path, device)
