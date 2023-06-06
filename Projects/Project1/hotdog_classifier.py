import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import logging
import os
import datetime
from sklearn.metrics import f1_score

# Our imports
from data_utils import get_dataLoader
from model.model_P import Network
from model.resnet18 import resnet18
from model.model_gradmod import Network_Grad_Mod

#Setup Device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists('logs'):
   os.makedirs('logs')

if not os.path.exists('trained_models'):
   os.makedirs('trained_models')

# Get model. Can be found in model folder
log_path = f"logs/training_{current_time}.log"
logging.basicConfig(filename=log_path, level=logging.INFO)

model = Network_Grad_Mod()
#model = resnet18()
model.to(device)

# Hyper parameters
learning_rate = 0.1
num_epochs = 20
batch_size = 64
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

trainset, testset, train_loader, test_loader = get_dataLoader(data_aug = True, batch_size = batch_size)

# Train function
def train(model, optimizer, scheduler, num_epochs=num_epochs):

    def loss_fun(output, target):
        target_one_hot = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy_with_logits(output, target_one_hot)

    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': [],
              'train_f1': [],
              'test_f1': []}
  
    for epoch in range(num_epochs):
        model.train()
        # For each epoch
        train_correct = 0
        train_loss = []
        train_labels, train_preds = [], []  # Collect true and predicted labels
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = loss_fun(output, target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
            train_labels.extend(target.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        # Step the learning rate scheduler
        scheduler.step()

        # Compute the test accuracy
        test_loss = []
        test_correct = 0
        test_labels, test_preds = [], []  # Collect true and predicted labels
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            test_labels.extend(target.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())

        # Compute F1 scores
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        test_f1 = f1_score(test_labels, test_preds, average='macro')

        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        out_dict['train_f1'].append(train_f1)
        out_dict['test_f1'].append(test_f1)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_{epoch+1}.pth')

        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%\t",
              f"F1 train: {out_dict['train_f1'][-1]*100:.1f}%\t test: {out_dict['test_f1'][-1]*100:.1f}%\t",
              f"Epoch: {epoch+1}/{num_epochs}")

        logging.info(f"Epoch: {epoch}, Loss train: {np.mean(train_loss):.3f}, Loss test: {np.mean(test_loss):.3f}, Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%, Accuracy test: {out_dict['test_acc'][-1]*100:.1f}%, F1 train: {out_dict['train_f1'][-1]*100:.1f}%, F1 test: {out_dict['test_f1'][-1]*100:.1f}%")

    logging.info(f"RUN FINISHED")
    return out_dict


#Train
out_dict = train(model, optimizer, scheduler)

#Save model
save_model_path = f"trained_models/our_model.pth"
torch.save(model.state_dict(), save_model_path)
