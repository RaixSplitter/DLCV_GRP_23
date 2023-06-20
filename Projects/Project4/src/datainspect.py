from selective_search import plot_images, generate_proposals_and_labels
from waste_dataset import WasteDatasetImages
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import torch
import os
import cv2


if __name__ == "__main__":
    DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    dataset = WasteDatasetImages(transform=transforms.ToTensor(), resize=(224, 224))
    
    #Slice dataset into a sample dataset with 10 images
    sample_dataset = Subset(dataset, range(10))

    #Create dataloader
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    sample_proposals, sample_data, sample_features, sample_images, sample_features_per_image = generate_proposals_and_labels(sample_loader, ss, len(sample_dataset), 100)
    
    sample_features_flat, sample_boxes = zip(*[(feature, bbox) for bbox, feature in sample_features])
    sample_features_flat = list(sample_features_flat)
    sample_boxes = list(sample_boxes)

    #Plot images
    for i, img in enumerate(sample_images):
        features_per_image = sample_features_per_image[i]
        bbox_list_by_image = []
        for feature in features_per_image:
            bbox, feature_hist = feature 
            bbox_list_by_image.append(bbox)
