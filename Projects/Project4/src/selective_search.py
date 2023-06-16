import json
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import matplotlib.patches as patches

from torchvision import transforms
import torchvision
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from waste_dataset import WasteDataset, WasteDatasetImages
import numpy as np


if __name__ == "__main__":

    DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    resize_dims = (512, 512)
    dataset = WasteDatasetImages(transform=transforms.ToTensor(), resize=resize_dims)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataloader)

    num_images_to_process = 10
    for idx, (image, bboxes) in enumerate(dataloader):
        if idx >= num_images_to_process:
            break

        img = image.squeeze().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots()
        ax.imshow(img)

        for bbox in bboxes:
            bbox = [item.item() for item in bbox]
            print(f"Bounding box: {bbox}")

            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        output_path = os.path.join(output_dir, f"image_{idx}.jpg")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Processed image {idx + 1}/{len(dataloader)}")

    print("Images with bounding boxes saved.")

    # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # num_images_to_process = 2

    # for idx, image in enumerate(dataloader):
    #     if idx >= num_images_to_process:
    #         break

    #     print(f"Processing image {idx + 1}/{num_images_to_process}")

    #     # Convert image from tensor to numpy array
    #     img = image.squeeze().permute(1, 2, 0).numpy()
    #     img = (img * 255).astype(np.uint8)  # Convert to uint8

    #     ss.setBaseImage(img)
    #     ss.switchToSelectiveSearchFast()
    #     rects = ss.process()
    #     print(f"Number of regions: {len(rects)}")
    #     print(f"Regions: {rects}")

    #     img_with_regions = img.copy()
    #     for i, (x, y, w, h) in enumerate(rects):
    #         cv2.rectangle(img_with_regions, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #     output_path = os.path.join(output_dir, f"image_{idx}.jpg")
    #     cv2.imwrite(output_path, img_with_regions)

    #     num_rects = len(rects)
    #     print(f"Processed image {idx + 1}/{num_images_to_process}")
    #     print(f"Number of rectangles: {num_rects}")

    # print("Selective search completed.")


