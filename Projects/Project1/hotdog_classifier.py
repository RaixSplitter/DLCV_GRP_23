from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from data_utils import get_dataLoader

train_loader, test_loader = get_dataLoader(data_aug = False, batch_size = 64)
print(train_loader)
print(test_loader)

