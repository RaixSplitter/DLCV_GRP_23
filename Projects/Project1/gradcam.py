from typing import Any, Mapping
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Our imports
from data_utils import get_dataLoader
from model.model_P import Network

data_loader = get_dataLoader(data_aug = False, batch_size = 64)

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Network_Grad_Mod(nn.Module):
    def __init__(self, weights_path : str):
        super(Network_Grad_Mod, self).__init__()
        self.model = Network()
        self.model.state_dict(torch.load(weights_path))
        
        


    

save_model_path = f"trained_models/our_model.pth"
model = Network_Grad_Mod(save_model_path)

print(model)    

#    def forward(self, x):