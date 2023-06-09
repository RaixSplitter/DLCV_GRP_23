import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt


class DRIVEDataset(Dataset):
    def __init__(self, img_dir, mask_dir, eye_mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        #self.eye_mask_dir = eye_mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))  # Sort the filenames
        self.masks = sorted(os.listdir(mask_dir))  # Sort the filenames
        #self.eye_masks = sorted(os.listdir(eye_mask_dir))  # Sort the filenames

    def __len__(self):
        # Assume that the number of images, masks, and eye masks are the same
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        #eye_mask_name = self.eye_masks[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        #eye_mask_path = os.path.join(self.eye_mask_dir, eye_mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        #eye_mask = Image.open(eye_mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            #eye_mask = self.transform(eye_mask)

        return image, mask, eye_mask


# Define a transform to normalize the data
tsfm = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert PIL image to tensor
])

def visualize(dataset, idx, save_dir):
    image, mask, eye_mask = dataset[idx]  # Get the items
    image = image.permute(1, 2, 0)  # Reorder dimensions to (H, W, C)
    
    fig, ax = plt.subplots(1, 2, figsize=(15,5))  # Create a subplot with 1 row and 3 columns

    ax[0].imshow(image)  # Display the image
    ax[0].set_title('Image')

    ax[1].imshow(mask.squeeze(), cmap='gray')  # Display the mask
    ax[1].set_title('Mask')
    
    for a in ax:
        a.axis('off')  # Turn off axis

    plt.savefig(os.path.join(save_dir, f'data_{idx}.png'))  # Save the figure


current_path = os.getcwd()
image_path = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'images')
vessel_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', '1st_manual')
eye_mask = os.path.join(current_path, '..', 'data', 'DRIVE', 'training', 'mask')
plot_path = os.path.join(current_path, '..', 'plots')


dataset = DRIVEDataset(image_path, vessel_mask, eye_mask, transform=tsfm)

train_size = int(0.7 * len(dataset))  # 70% of dataset for training
val_size = int(0.15 * len(dataset))  # 15% of dataset for validation
test_size = len(dataset) - train_size - val_size  # remaining for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create dataloaders for each dataset
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print('Loaded %d training images' % len(train_dataset))
print('Loaded %d val images' % len(val_dataset))
print('Loaded %d test images' % len(test_dataset))

visualize(train_dataset, 1, plot_path)