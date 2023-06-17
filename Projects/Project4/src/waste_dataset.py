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
        bounding_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        subimage = src_img.crop(bounding_box)

        return self.transform(subimage), label

class WasteDatasetImages(Dataset):
    def __init__(self, transform=None, resize=None):
        with open(os.path.join(DATA_PATH, 'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_info = data['images']
        self.annotation = data['annotations']
        self.resize = resize  # Specify the desired resize dimensions
    
    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        img_info = self.img_info[idx]
        img_id = img_info['id']
        src_img_file = img_info['file_name']
        src_img = Image.open(os.path.join(DATA_PATH, src_img_file))
        
        # Resize the image
        resized_img = src_img.resize(self.resize)
        transformed_img = self.transform(resized_img)
        
        bboxes = [ann['bbox'] for ann in self.annotation if ann['image_id'] == img_id]
        #print(f"Found {len(bboxes)} bounding boxes for image {img_id}")
        resized_bboxes = []
        for bbox in bboxes:
            resized_bbox = [
                bbox[0] * self.resize[0] / src_img.width,  # x
                bbox[1] * self.resize[1] / src_img.height,  # y
                bbox[2] * self.resize[0] / src_img.width,  # width
                bbox[3] * self.resize[1] / src_img.height  # height
            ]
            resized_bboxes.append(resized_bbox)

        return transformed_img, resized_bboxes

