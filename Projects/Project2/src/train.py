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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Paths
current_path = os.getcwd()
image_path = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'images')
vessel_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', '1st_manual')
eye_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'mask')
plot_path = os.path.join(current_path, '..', 'plots')
results_path = os.path.join(current_path, '..', 'results')

size = 128 # Image size 
train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = get_data_loader(image_path, vessel_mask, size)

print('Loaded %d training images' % len(train_dataset))
print('Loaded %d val images' % len(val_dataset))
print('Loaded %d test images' % len(test_dataset))

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

learning_rate = 0.001
epochs = 20

def train(model, opt, loss_fn, epochs, train_loader, test_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(2):  # change this to change the number of images shown
            plt.subplot(2, 2, k+1)
            plt.imshow(np.transpose(X_test[k].numpy(), (1, 2, 0)), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 2, k+3)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_results.png"))  # Save the figure
        plt.clf()  # Clear the current figure for the next plot


model = EncDec().to(device)

train(model, optim.Adam(model.parameters(), lr=learning_rate), bce_loss, epochs, train_dataloader, val_dataloader, results_path)