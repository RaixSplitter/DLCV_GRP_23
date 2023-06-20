import torch
import os
import numpy as np
import json
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt

DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"
ALL_SUPERCATEGORIES = ["Background", "Aluminium foil", "Bottle", "Bottle cap", "Broken glass", "Can", "Carton", "Cup", "Lid", "Other plastic", "Paper", "Plastic bag & wrapper", "Plastic container", "Pop tab", "Straw", "Styrofoam piece", "Unlabeled litter", "Cigarette", "Plastic utensils", "Rope & strings", "Paper bag", "Scrap metal", "Food waste", "Shoe", "Squeezable tube", "Blister pack", "Glass jar", "Plastic glooves", "Battery"]
#SUPERCATEGORIES = ["Background", "Bottle", "Bottle cap", "Can", "Carton", "Plastic bag & wrapper", "Unlabeled litter", "Cup", "Straw", "Paper", "Broken glass"]
SUPERCATEGORIES = ALL_SUPERCATEGORIES
UNLABELED = SUPERCATEGORIES.index("Unlabeled litter")

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
        self.categories = data['categories']
        self.cat_size = {}
        for item in self.annotation:
            supercat = self.categories[item['category_id']]['supercategory']
            #if we're not using the class (not in supercategories list), set to unlabeled
            label = SUPERCATEGORIES.index(supercat) if supercat in SUPERCATEGORIES else UNLABELED
            if label in self.cat_size.keys():
                self.cat_size[label] += 1
            else: self.cat_size[label] = 0
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        item = self.annotation[idx]
        #assert idx+1 == item['id']

        src_img_data = self.img_info[item['image_id']]
        assert src_img_data['id'] == item['image_id']

        src_img = Image.open(os.path.join(DATA_PATH,src_img_data['file_name']))

        supercat = self.categories[item['category_id']]['supercategory']
        #if we're not using the class (not in supercategories list), set to unlabeled
        label = SUPERCATEGORIES.index(supercat) if supercat in SUPERCATEGORIES else UNLABELED
        bbox = item['bbox']
        bounding_box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        subimage = src_img.crop(bounding_box)

        return self.transform(subimage), label
    
    def num_categories(self):
        return len(SUPERCATEGORIES)
    
    def category_name(self, label):
        return SUPERCATEGORIES[label]
    
    def category_size(self, label):
        return self.cat_size[label] if label in self.cat_size.keys() else 0
    
    def save_category_count_plot(self):
        sorted_sizes = sorted(self.cat_size.items(), key=lambda x: x[1], reverse=False)
        keys = [SUPERCATEGORIES[item[0]] for item in sorted_sizes]
        values = [item[1] for item in sorted_sizes]
        plt.barh(keys, values)
        plt.tight_layout()
        plt.savefig('bars.png')

class WasteDatasetImages(Dataset):
    def __init__(self, transform=None, resize=(224,224)):
        with open(os.path.join(DATA_PATH, 'annotations.json')) as f:
            data = json.load(f)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor()
        ])
        self.img_info = data['images']
        self.annotation = data['annotations']
        self.categories = data['categories']
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
            supercat = self.categories[patch['category_id']]['supercategory']
            #if we're not using the class (not in supercategories list), set to unlabeled
            label = SUPERCATEGORIES.index(supercat) if supercat in SUPERCATEGORIES else UNLABELED
            labels.append(label)

        return transformed_img, resized_bboxes, labels
    
    def num_categories(self):
        return len(SUPERCATEGORIES)
    
    def category_name(self, label):
        return SUPERCATEGORIES[label]

