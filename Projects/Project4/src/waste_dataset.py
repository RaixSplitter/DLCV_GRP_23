import torch
import os
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"

class WasteDatasetPatches(Dataset):
    def __init__(self, transform=None, resolution=(64,64)):
        with open(os.path.join(DATA_PATH,'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor()
            ])
        self.img_info = data['images']
        self.annotation = data['annotations']
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        item = self.annotation[idx]
        #assert idx+1 == item['id']

        src_img_data = self.img_info[item['image_id']]
        assert src_img_data['id'] == item['image_id']

        src_img = Image.open(os.path.join(DATA_PATH,src_img_data['file_name']))
        src_img.save('original_img.png')

        label = item['category_id']
        bbox = item['bbox']
        bounding_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        subimage = src_img.crop(bounding_box)

        return self.transform(subimage), label

class WasteDatasetImages(Dataset):
    def __init__(self, transform=None, img_size=512):
        with open(os.path.join(DATA_PATH, 'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
            ])
        self.img_info = data['images']
        self.annotation = data['annotations']
        self.img_size = img_size
    
    def __len__(self):
        return len(self.img_info)
    
    def __getitem__(self, idx):
        img_info = self.img_info[idx]
        img_id = img_info['id']
        src_img_file = img_info['file_name']
        src_img = Image.open(os.path.join(DATA_PATH, src_img_file))
        
        # Apply transform
        transformed_img = self.transform(src_img)
        
        bboxes = [ann['bbox'] for ann in self.annotation if ann['image_id'] == img_id][0]
        #print(f"Found {len(bboxes)} bounding boxes for image {img_id}")
        bboxes[0] = int(bboxes[0] * self.img_size / src_img.width)  # x
        bboxes[1] = int(bboxes[1] * self.img_size / src_img.height) # y
        bboxes[2] = int(bboxes[2] * self.img_size / src_img.width)  # width
        bboxes[3] = int(bboxes[3] * self.img_size / src_img.height) # height

        return transformed_img, torch.Tensor(bboxes)

