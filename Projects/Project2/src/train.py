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
from torch.optim.lr_scheduler import StepLR
import logging
import datetime

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
            plt.imshow(Y_hat[k, 0], cmap='RGB')
            plt.title('Output')
            plt.axis('off')
        
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_results.png"))  # Save the figure
        plt.clf()  # Clear the current figure for the next plot


def train_with_metrics(model, opt, loss_fn, epochs, train_loader, test_loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    X_test, Y_test = next(iter(test_loader))

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"../logs/training_{current_time}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO)

    scheduler = StepLR(opt, step_size=25, gamma=0.1)  # Define the scheduler with appropriate parameters

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        avg_dice_train, avg_iou_train, avg_acc_train, avg_sens_train, avg_spec_train = 0, 0, 0, 0, 0
        avg_dice_test, avg_iou_test, avg_acc_test, avg_sens_test, avg_spec_test = 0, 0, 0, 0, 0
        
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

            # calculate metrics for training set
            avg_loss += loss / len(train_loader)
            Y_pred_bin = (Y_pred > 0.5).type(torch.int)
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
            avg_dice_train += dice_score / len(train_loader)
            avg_iou_train += iou_score / len(train_loader)
            avg_acc_train += accuracy / len(train_loader)
            avg_sens_train += sensitivity / len(train_loader)
            avg_spec_train += specificity / len(train_loader)

        # calculate metrics for test set
        model.eval()  # testing mode
        with torch.no_grad():
            test_loss = 0
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                Y_pred = model(X_batch)
                loss = loss_fn(Y_batch, Y_pred)  # forward-pass
                test_loss += loss.item() / len(test_loader)
                Y_pred_bin = (Y_pred > 0.5).type(torch.int)
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
                avg_dice_test += dice_score / len(test_loader)
                avg_iou_test += iou_score / len(test_loader)
                avg_acc_test += accuracy / len(test_loader)
                avg_sens_test += sensitivity / len(test_loader)
                avg_spec_test += specificity / len(test_loader)

        print(' - Train Loss: %.4f, Dice: %.2f, IoU: %.2f, Acc: %.2f, Sens: %.2f, Spec: %.2f' % (avg_loss, avg_dice_train, avg_iou_train, avg_acc_train, avg_sens_train, avg_spec_train))
        print(' - Test Loss: %.4f, Dice: %.2f, IoU: %.2f, Acc: %.2f, Sens: %.2f, Spec: %.2f' % (test_loss, avg_dice_test, avg_iou_test, avg_acc_test, avg_sens_test, avg_spec_test))

        logging.info(f"Epoch: {epoch}, Train Loss: {avg_loss:.4f}, Dice: {avg_dice_train:.2f}, IoU: {avg_iou_train:.2f}, Acc: {avg_acc_train:.2f}, Sens: {avg_sens_train:.2f}, Spec: {avg_spec_train:.2f}")
        logging.info(f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Dice: {avg_dice_test:.2f}, IoU: {avg_iou_test:.2f}, Acc: {avg_acc_test:.2f}, Sens: {avg_sens_test:.2f}, Spec: {avg_spec_test:.2f}")
        
        scheduler.step()

        # show intermediate results
        model.eval()  # testing mode
        Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(2):  # change this to change the number of images shown
            plt.subplot(2, 3, 3*k+1)
            plt.imshow(np.transpose(X_test[k].numpy(), (1, 2, 0)), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 3, 3*k+2)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
            
            plt.subplot(2, 3, 3*k+3)
            plt.imshow(Y_test[k, 0], cmap='gray')
            plt.title('Target')
            plt.axis('off')

        plt.suptitle('Result')
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch+1}_results.png"))  # Save the figure
        plt.clf()  # Clear the current figure for the next plot


