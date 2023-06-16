import torch
import torch.nn as nn
import torch.nn.functional as F

class UNeet(nn.Module):
    def __init__(self, in_ch = 3):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(                   ConvBlock(in_ch,  64, 3, padding=1)) 
        self.enc_conv1 = nn.Sequential(nn.MaxPool2d(2, 2),ConvBlock(   64, 128, 3, padding=1)) 
        self.enc_conv2 = nn.Sequential(nn.MaxPool2d(2, 2),ConvBlock(  128, 256, 3, padding=1)) 
        self.enc_conv3 = nn.Sequential(nn.MaxPool2d(2, 2),ConvBlock(  256, 512, 3, padding=1)) 

        # bottleneck
        self.bottleneck_conv = nn.Sequential(nn.MaxPool2d(2, 2),
                                             nn.Conv2d( 512, 1024, 3, padding=1), 
                                             nn.Conv2d(1024, 1024, 3, padding=1), 
                                             nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # -> [_,  /2, _, _]
                                            )

        # decoder (upsampling)
        self.upsample0 = nn.Sequential(ConvBlock(1024, 512, 3, padding=1), nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.upsample1 = nn.Sequential(ConvBlock( 512, 256, 3, padding=1), nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.upsample2 = nn.Sequential(ConvBlock( 256, 128, 3, padding=1), nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2))
        self.upsample3 = nn.Sequential(ConvBlock( 128,  64, 3, padding=1), nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x )                     # -> [2,  64, 256, 256]
        e1 = self.enc_conv1(e0)                     # -> [2, 128, 128, 128]
        e2 = self.enc_conv2(e1)                     # -> [2, 256,  64,  64]
        e3 = self.enc_conv3(e2)                     # -> [2, 512,  32,  32]

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))        # -> [2, 512,  32,  32]

        #    torch.cat([n1,n2], 1).shape              -> [_,  2x,   _,   _]
        # decoder
        d0 = self.upsample0(torch.cat([b ,e3], 1))  # -> [2, 256,  64,  64]
        d1 = self.upsample1(torch.cat([d0,e2], 1))  # -> [2, 128, 128, 128]
        d2 = self.upsample2(torch.cat([d1,e1], 1))  # -> [2,  64, 256, 256]
        d3 = self.upsample3(torch.cat([d2,e0], 1))  # -> [2,   1, 256, 256]
        return d3

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, kernel_size = kernel_size, padding = padding), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_channels),
            
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding), 
            nn.ReLU(), 
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.block(x)