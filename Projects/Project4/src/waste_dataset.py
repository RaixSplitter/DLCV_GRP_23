import torch
import os
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"

class WasteDataset(Dataset):
    def __init__(self, transform=None):
        with open(os.path.join(DATA_PATH,'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor()
            ])
        self.img_info = data['images']
        self.annotation = data['annotations']
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        item = self.annotation[idx]
        src_img_file = self.img_info[item['image_id']]['file_name']
        src_img = Image.open(os.path.join(DATA_PATH,src_img_file))

        label = item['category_id']
        bbox = item['bbox']
        bounding_box = [min(bbox[0],bbox[2]), min(bbox[1],bbox[3]), max(bbox[0],bbox[2]), max(bbox[1],bbox[3])]
        subimage = src_img.crop(bounding_box)

        return self.transform(subimage), label
