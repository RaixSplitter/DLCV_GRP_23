from typing import Any, Mapping
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torchvision.models as models

# Our imports
from model.model_gradmod import Network_Grad_Mod as Network
#from model_gradmod import Network_Grad_Mod as Network


class Network_Grad_Mod(nn.Module):
    def __init__(self, weights_path : str):
        super(Network_Grad_Mod, self).__init__()
        self.model = Network()
        self.model.state_dict(torch.load(weights_path))

        self.features_conv = self.model.features[:15]

        self.batch_norm = nn.BatchNorm2d(32) 
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.classifier = self.model.fully_connected

        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features_conv(x)
    
class Resnet_Grad_Mod(nn.Module):
    def __init__(self, weights_path : str):
        super(Resnet_Grad_Mod, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        try:
            self.model.load_state_dict(torch.load(weights_path))
        except RuntimeError:
            pass
        self.features_conv = nn.Sequential(
            self.model.layer4[0],
            self.model.layer4[1].conv1,
            self.model.layer4[1].bn1,
            self.model.layer4[1].relu,
            self.model.layer4[1].conv2,
        )

        self.batch_norm = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.features_conv(x)
        h = x.register_hook(self.activations_hook)
        x = self.batch_norm(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.model.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.features_conv(x)
        return x

        
        
if __name__ == '__main__':
    save_model_path = f"../trained_models/resnet.pth"
    model = Resnet_Grad_Mod(save_model_path)
    a = Network()
    print(a)
    #print(model)
    print(model.model.fc)
    # print(model.model.layer4[1])
    # print(model.model.layer4[1].conv2)

        