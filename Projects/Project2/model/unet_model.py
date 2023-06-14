import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class DownConv(nn.module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.maxpool_doubble_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
            )
        
    def forward(self, x):
        return self.maxpool_doubble_conv(x)
    
