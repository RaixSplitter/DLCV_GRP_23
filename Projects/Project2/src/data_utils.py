import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt


class DRIVEDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))  # Sort the filenames
        self.masks = sorted(os.listdir(mask_dir))  # Sort the filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def visualize(dataset, idx, save_dir):
    image, mask, eye_mask = dataset[idx] 
    image = image.permute(1, 2, 0) 
    
    fig, ax = plt.subplots(1, 2, figsize=(15,5))

    ax[0].imshow(image)
    ax[0].set_title('Image')

    ax[1].imshow(mask.squeeze(), cmap='gray')
    ax[1].set_title('Mask')
    
    for a in ax:
        a.axis('off')

    plt.savefig(os.path.join(save_dir, f'data_{idx}.png'))  # Save the figure


def get_data_loader(image_path, vessel_mask, size):

    tsfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    dataset = DRIVEDataset(image_path, vessel_mask, transform=tsfm)

    train_size = int(0.7 * len(dataset))  # 70% of dataset for training
    # val_size = int(0.15 * len(dataset))  # 15% of dataset for validation
    val_size = int(0.3 * len(dataset))  # 30% of dataset for validation
    test_size = len(dataset) - train_size - val_size  # remaining for testing

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # raise Exception(len(train_dataset), len(val_dataset), len(test_dataset))
    # Create dataloaders for each dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
