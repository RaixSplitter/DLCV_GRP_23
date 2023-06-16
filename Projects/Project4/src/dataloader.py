# import os
import glob
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# with open('/dtu/datasets1/02514/data_wastedetection/annotations.json') as json_file:
#     jdata = json.load(json_file)
# print(jdata) # useless keys: "info","images",

class waste_dataset(Dataset):
    def __init__(self, transform = None, img_size = 400):
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor()
            ])
        self.img_paths = glob.glob("/dtu/datasets1/02514/data_wastedetection/**/*.JPG")
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        label = None
        X = self.transform(image)
        Y = self.transform(label)
        return X, Y
    
def get_dataloaders(img_size=400):
    data = waste_dataset(img_size=img_size)
    train_size = int(0.75*len(data))
    test_size = int(0.15*len(data))
    val_size = len(data) - train_size - test_size

    train, test, val = random_split(data, [train_size, test_size, val_size])
    train_dl = DataLoader(train, batch_size=16, shuffle=True)
    test_dl = DataLoader(test, batch_size=16, shuffle=False)
    val_dl = DataLoader(val, batch_size=16, shuffle=False)
    
    return train_dl, test_dl, val_dl