import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, image_size : int = 128):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024)
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 8 -> 16
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512)
        )
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 16 -> 32
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256)
        )
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 32 -> 64
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        p0 = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(p0))
        p1 = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(p1))
        p2 = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(p2))
        p3 = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(p3))

        # decoder
        u0 = self.upsample0(b)
        d0 = torch.cat([u0, e3], dim=1)
        d0 = F.relu(self.dec_conv0(d0))
        u1 = self.upsample1(d0)
        d1 = torch.cat([u1, e2], dim=1)
        d1 = F.relu(self.dec_conv1(d1))
        u2 = self.upsample2(d1)
        d2 = torch.cat([u2, e1], dim=1)
        d2 = F.relu(self.dec_conv2(d2))
        u3 = self.upsample3(d2)
        d3 = torch.cat([u3, e0], dim=1)
        d3 = self.dec_conv3(d3)  # no activation

        return d3

from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
summary(model, (3, 128, 128))