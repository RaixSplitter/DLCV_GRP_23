import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def resnet18():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft
