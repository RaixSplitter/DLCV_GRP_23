import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network_Grad_Mod(nn.Module):
    def __init__(self):
        super(Network_Grad_Mod, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            
        
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
        
        self.fully_connected = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(28*28*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2),
            )
        
    def forward(self, x):
      #reshaping x so it becomes flat, except for the first dimension (which is the minibatch)
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fully_connected(x)
        return x