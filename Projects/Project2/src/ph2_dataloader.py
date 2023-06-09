import torch
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

class ph2_dataset(Dataset):
    def __init__(self, transform = None):
        self.path = os.path.join(os.getcwd(),'data','PH2_Dataset_images')
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((400,400)),
            transforms.ToTensor()
            ])
        self.items = os.listdir(self.path)
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        item_name = self.items[index]
        item_path = os.path.join(self.path, item_name)
        image = Image.open(os.path.join(item_path, f'{item_name}_Dermoscopic_Image', f'{item_name}.bmp'))
        label = Image.open(os.path.join(item_path, f'{item_name}_lesion', f'{item_name}_lesion.bmp'))
        X = self.transform(image)
        Y = self.transform(label)
        return X, Y