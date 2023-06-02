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

# Our imports
from data_utils import get_dataLoader
from model.model_1 import Network

#Setup Device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('logs'):
   os.makedirs('logs')

if not os.path.exists('trained_models'):
   os.makedirs('trained_models')

# Get model. Can be found in model folder
log_path = "logs/training.log"
logging.basicConfig(filename=log_path, level=logging.INFO)

model = Network()
model.to(device)

# Hyper parameters
learning_rate = 0.001
num_epochs = 2
batch_size = 64
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainset, testset, train_loader, test_loader = get_dataLoader(data_aug = False, batch_size = batch_size)

# Train function
def train(model, optimizer, num_epochs=num_epochs):

    def loss_fun(output, target):
        target_one_hot = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy_with_logits(output, target_one_hot)

    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
        logging.info(f"Epoch: {epoch}, Loss train: {np.mean(train_loss):.3f}, Loss test: {np.mean(test_loss):.3f}, Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%, Accuracy test: {out_dict['test_acc'][-1]*100:.1f}%")
    logging.info(f"RUN FINISHED")
    return out_dict

#Train
out_dict = train(model, optimizer)

#Save model
save_model_path = f"trained_models/Network_model.pth"
torch.save(model.state_dict(), save_model_path)
