import os
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from torchvision.transforms import functional as F
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Own
from waste_dataset import WasteDatasetImages
from classifier.resnet18 import resnet18, resnet18_inference
from R_CNN import* 
from utility import*
    # %%
    # Main

def plot_image_with_boxes(image, bboxes, labels, title, filename):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=colors[i % len(colors)], facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, str(label), color=colors[i % len(colors)])

    plt.title(title)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    ctest = 0
    ctrain = 0
    patch_size = (64,64)
    batch_size = 32

    dataset = WasteDatasetImages(transform=transforms.ToTensor(), resize=(256,256))
    num_classes = dataset.num_categories()
    print(f"Total number of categories: {num_classes}")

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    max_proposals_per_image = 1000 # Selective search will generate max 1000 proposals per image
    num_images_to_process_train = int(len(train_dataset)/2) #Amount of train images to process
    num_images_to_process_test = 20 #Amount of test images to process

    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    real_bbox_label_pairs = []
    real_bbox_label_image_vis = []  # for visualization purposes

    predicted_bbox_label_vis = []  # for visualization purposes


    for batch in test_dataloader:
        if ctest >= num_images_to_process_test:
            break
        images, bboxes, labels = batch
        for bbox, label in zip(bboxes, labels):
            bbox = [coord.item() for coord in bbox]  # Convert tensor to scalar
            real_bbox_label_pairs.append([bbox, label.item()])
            
            # For visualization
            if ctest < 20:  # Only for the first 5 images
                real_bbox_label_image_vis.append({
                    "image": images[0].numpy(),  # Convert tensor to numpy array
                    "bbox": bbox,
                    "label": label.item()
                })
        ctest += 1

    # Run selective search 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    print("inference")

    model = resnet18_inference(8)  # Replace with your actual model class
    model.load_state_dict(torch.load("trained_models/resnet18_model_10.pth"))
    model.eval()


    _, proposals_box_list, resized_images, _, _, proposals_per_image = generate_proposals_and_labels(test_dataloader, ss, num_images_to_process_test, max_proposals_per_image)

    resized_images_array = np.array(resized_images, dtype=np.float32)
    resized_images_tensor = torch.from_numpy(resized_images_array)

    raw_scores = model(resized_images_tensor)

    # Convert raw scores to probabilities
    probabilities = F.softmax(raw_scores, dim=1)
    #print("probabilities", probabilities)

    predicted_labels = torch.argmax(probabilities, dim=1)

    bbox_label_pairs = []
    for i, (bbox, label) in enumerate(zip(proposals_box_list, predicted_labels.tolist())):
        if label != 0:
            probability = probabilities[i][label].item()
            bbox_label_pairs.append([bbox, label, probability])

    predicted_bbox_label_vis_per_image = []  # Store the predicted bounding boxes and labels per image
    current_index = 0

    for img_bboxes in proposals_per_image:
        img_predictions = []
        for bbox in img_bboxes:
            label = predicted_labels[current_index].item()
            if label != 0:
                probability = probabilities[current_index][label].item()
                img_predictions.append([bbox, label])
            current_index += 1
        predicted_bbox_label_vis_per_image.append(img_predictions)

    print("Real bounding box and label pairs for visualization: ", real_bbox_label_image_vis)
    print("Predicted bounding box and label pairs for visualization: ", predicted_bbox_label_vis_per_image)

    print("bbox before no max", bbox_label_pairs)

    bbox_new = no_max_supression(bbox_label_pairs, 0.5)
    print("bbox after no max", bbox_new)
    average_precision, precision, recall = mean_average_precision(real_bbox_label_pairs, bbox_new)
    print("average_precision", average_precision)
    print("precision", precision)
    print("recall", recall)

    os.makedirs("plots", exist_ok=True)
    for idx, (real_data, pred_data) in enumerate(zip(real_bbox_label_image_vis, predicted_bbox_label_vis_per_image)):
        image = real_data["image"].transpose(1, 2, 0)  # Transpose the image to the HWC format

        # Unpack the real bounding boxes and labels
        real_bboxes = [real_data["bbox"]]
        real_labels = [real_data["label"]]

        # Unpack the predicted bounding boxes and labels
        pred_bboxes = [pred[0] for pred in pred_data]
        pred_labels = [pred[1] for pred in pred_data]

        # Plot and save the image with the real and predicted bounding boxes
        plot_image_with_boxes(image, real_bboxes, real_labels, "Real", f"plots/real_{idx}.png")
        plot_image_with_boxes(image, pred_bboxes, pred_labels, "Predicted", f"plots/predicted_{idx}.png")