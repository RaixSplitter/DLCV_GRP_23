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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from waste_dataset import WasteDatasetPatches
from simpleCNN import SimpleClassifier
from resnet18 import resnet18

#Setup Device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, learning_rate, epochs, loss_func, train_dl, test_dl, val_dl, train_size):
    model.to(device)
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': [],
              'train_f1': [],
              'test_f1': []}
    
    for epoch in range(epochs):
        if epoch%10 == 0: 
            learning_rate = learning_rate*0.1
            optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
        model.train()
        train_correct = 0
        train_loss = []
        train_labels, train_preds = [], []  # Collect true and predicted labels
        for idx, (data, target) in enumerate(train_dl):
            print(f'Start batch {idx}/{len(train_dl)} with {len(data)}')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            train_labels.extend(target.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        # Compute the test accuracy
        test_loss = []
        test_correct = 0
        test_labels, test_preds = [], []  # Collect true and predicted labels
        model.eval()
        for data, target in test_dl:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_func(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            test_labels.extend(target.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

        # Compute F1 scores
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        out_dict['train_acc'].append(train_correct/train_size)
        out_dict['test_acc'].append(test_correct/train_size)
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        out_dict['train_f1'].append(train_f1)
        out_dict['test_f1'].append(test_f1)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join('trained_models',f'resnet18_model_{epoch+1}.pth'))

        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
              f"F1 train: {out_dict['train_f1'][-1]*100:.1f}%\t test: {out_dict['test_f1'][-1]*100:.1f}%\t",
              f"Epoch: {epoch+1}/{epochs}")
        
        logging.info(f"Epoch: {epoch}, Loss train: {np.mean(train_loss):.3f}, Loss test: {np.mean(test_loss):.3f}, Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%, Accuracy test: {out_dict['test_acc'][-1]*100:.1f}%, F1 train: {out_dict['train_f1'][-1]*100:.1f}%, F1 test: {out_dict['test_f1'][-1]*100:.1f}%")

    torch.save(model.state_dict(), os.path.join('trained_models',f'resnet18_final_model.pth'))
    logging.info(f"RUN FINISHED")
    return out_dict

if __name__ == '__main__':
    patch_size = (64,64)
    data = WasteDatasetPatches(resolution=patch_size)
    train_size = int(0.60*len(data))
    test_size = int(0.20*len(data))
    val_size = len(data) - train_size - test_size

    train_ds, test_ds, val_ds = random_split(data, [train_size, test_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)

    #network = SimpleClassifier(num_classes=data.num_categories(), resolution=patch_size)
    network = resnet18(data.num_categories())
    if len(sys.argv) > 1:
        try:
            network.load_state_dict(torch.load(f'./trained_models/{sys.argv[1]}.pth'))
        except:
            print(f'no network saved at trained_models/{sys.argv[1]}.pth')
    network.to(device)
    learning_rate = 0.01
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    loss = F.cross_entropy

    train(network, optimizer, learning_rate, 25, loss, train_dl, test_dl, val_dl, len(train_ds))
