import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.encoder0 = nn.Sequential(ConvBlock(in_channels, 64))
        self.encoder1 = nn.Sequential(nn.MaxPool2d(2,2), ConvBlock(64, 128))
        self.encoder2 = nn.Sequential(nn.MaxPool2d(2,2), ConvBlock(128, 256))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2,2), ConvBlock(256,512), nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.decoder0 = nn.Sequential(ConvBlock(512,256), nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.decoder1 = nn.Sequential(ConvBlock(256,128), nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.decoder2 = nn.Sequential(ConvBlock(128,64), nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1))

    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.bottleneck(x2)
        x3 = self.decoder0(torch.cat([x2,x3],dim=1))
        x3 = self.decoder1(torch.cat([x1,x3],dim=1))
        x3 = self.decoder2(torch.cat([x0,x3],dim=1))

        return x3
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)