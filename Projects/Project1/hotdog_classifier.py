from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from data_utils import get_dataLoader

train_data, test_data = get_dataLoader(data_aug = False)
print(train_data)
print(test_data)

