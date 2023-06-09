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

import matplotlib.pyplot as plt
from IPython.display import clear_output

from data_utils import DRIVEDataset, get_data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Paths
current_path = os.getcwd()
image_path = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'images')
vessel_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', '1st_manual')
eye_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'mask')
plot_path = os.path.join(current_path, '..', 'plots')

size = 128 # Image size 
train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = get_data_loader(image_path, vessel_mask, size)

print('Loaded %d training images' % len(train_dataset))
print('Loaded %d val images' % len(val_dataset))
print('Loaded %d test images' % len(test_dataset))


def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
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
        toc = time()
        print(' - loss: %f' % avg_loss)

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()