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

def train(model, opt, loss_fn, epochs, train_loader, test_loader, save_dir, device):

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

def train_with_metrics(model, opt, loss_fn, epochs, train_loader, test_loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        avg_dice, avg_iou, avg_acc, avg_sens, avg_spec = 0, 0, 0, 0, 0
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

            # binary predictions
            Y_pred_bin = (Y_pred > 0.5).type(torch.int)

            # compute metrics
            intersection = torch.logical_and(Y_batch, Y_pred_bin)
            union = torch.logical_or(Y_batch, Y_pred_bin)
            iou_score = torch.sum(intersection) / torch.sum(union)

            dice_score = 2. * torch.sum(intersection) / (torch.sum(Y_batch) + torch.sum(Y_pred_bin))

            tn = ((Y_pred_bin == 0) & (Y_batch == 0)).sum()
            tp = ((Y_pred_bin == 1) & (Y_batch == 1)).sum()
            fn = ((Y_pred_bin == 0) & (Y_batch == 1)).sum()
            fp = ((Y_pred_bin == 1) & (Y_batch == 0)).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            avg_dice += dice_score / len(train_loader)
            avg_iou += iou_score / len(train_loader)
            avg_acc += accuracy / len(train_loader)
            avg_sens += sensitivity / len(train_loader)
            avg_spec += specificity / len(train_loader)

        print(' - loss: %f, Dice: %f, IoU: %f, Acc: %f, Sens: %f, Spec: %f' % (avg_loss, avg_dice, avg_iou, avg_acc, avg_sens, avg_spec))

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
        
        plt.suptitle('result')
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_results.png"))  # Save the figure
        plt.clf()  # Clear the current figure for the next plot
