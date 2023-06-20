import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def resnet50(num_classes):
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft
