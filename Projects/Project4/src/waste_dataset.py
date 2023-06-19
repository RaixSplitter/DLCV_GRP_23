import torch
import os
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"
SUPERCATEGORIES = {"Aluminium foil":0, "Battery":1, "Blister pack":2, "Bottle":3, "Bottle cap":4, "Broken glass":5, "Can":6, "Carton":7, "Cup":8, "Food waste":9, "Glass jar":10, "Lid":11, "Other plastic":12, "Paper":13, "Paper bag":14, "Plastic bag & wrapper":15, "Plastic container":16, "Plastic glooves":17, "Plastic utensils":18, "Pop tab":19, "Rope & strings":20, "Scrap metal":21, "Shoe":22, "Squeezable tube":23, "Straw":24, "Styrofoam piece":25, "Unlabeled litter":26, "Cigarette":27}
CATEGORY_RANGE = [0,1,3,6,8,9,12,19,24,25,26,28,29,33,35,42,47,48,49,50,51,52,53,54,56,57,58,59]

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
        
        patches = [ann for ann in self.annotation if ann['image_id'] == img_id]
        #print(f"Found {len(bboxes)} bounding boxes for image {img_id}")
        resized_bboxes = []
        labels = []
        for patch in patches:
            bbox = patch['bbox']
            resized_bbox = [
                bbox[0] * self.resize[0] / src_img.width,  # x
                bbox[1] * self.resize[1] / src_img.height,  # y
                bbox[2] * self.resize[0] / src_img.width,  # width
                bbox[3] * self.resize[1] / src_img.height  # height
            ]
            resized_bboxes.append(resized_bbox)
            subclass = patch['category_id']
            label = next((idx for idx, x in enumerate(CATEGORY_RANGE) if subclass<=x), None)
            labels.append(label)

        return transformed_img, resized_bboxes, labels

