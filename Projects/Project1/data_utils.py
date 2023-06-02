from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataLoader(data_aug = False):

    current_path = os.getcwd()
    train_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'train')
    test_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'test')

    train_data = datasets.ImageFolder(train_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)
    
    if data_aug == True:
        transform = transforms.Compose([
        transforms.Resize((224,224)), # resizing to same size, for example 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization settings for RGB
    ])
    else: 
        transform = transforms.Compose([
        transforms.Resize((224,224)), # resizing to same size, for example 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization settings for RGB
    ])


    return train_data, test_data