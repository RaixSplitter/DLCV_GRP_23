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
from torch.utils.data import DataLoader, random_split
from sklearn.svm import LinearSVC
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump, load
from torchvision import models
import torch



def plot_images(dataloader):
    num_images_to_process = 10
    for idx, (image, bboxes) in enumerate(dataloader):
        if idx >= num_images_to_process:
            break

        img = image.squeeze().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots()
        ax.imshow(img)

        for bbox in bboxes:
            bbox = [item.item() for item in bbox]
            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        output_path = os.path.join(output_dir, f"image_{idx}.jpg")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Processed image {idx + 1}/{len(dataloader)}")

    print("Images with bounding boxes saved.")

def draw_bboxes(image, bboxes, labels, index):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch
    for bbox, label in zip(bboxes, labels):
        x, y, width, height = bbox
        rect = patches.Rectangle((x,y), width, height, linewidth=1, edgecolor='r' if label == 1 else 'b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    if not os.path.exists("inference"):
        os.makedirs("inference")

    plt.axis('off')
    plt.savefig(f'inference/(bbox_visualization_on_test_images_{index}.png')
    plt.close(fig)



def visualize_bboxes(image, proposals, ground_truth_bboxes, iou_scores):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox in ground_truth_bboxes:
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h,
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    for proposal, iou_score in zip(proposals, iou_scores):
        proposal_x, proposal_y, proposal_w, proposal_h = proposal

        if iou_score > 0.2:
            rect = patches.Rectangle((proposal_x, proposal_y), proposal_w, proposal_h,
                                 linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            ax.text(proposal_x, proposal_y, f"IOU: {float(iou_score):.2f}", color='r')

    plt.axis('off')
    plt.savefig('bbox_visualization.png')
    plt.close(fig)

def generate_proposals_and_labels(dataloader, ss, num_images_to_process, max_proposals_per_image):
    proposals_list = []
    data_list = []
    images_list = []  # Add this line to keep the images
    features_list = []  # Add this line

    counter = 0

    for image, bboxes in dataloader:
        print(f"Processing image {counter + 1}/{num_images_to_process}")
        if counter >= num_images_to_process:
            break
        img = image.squeeze().permute(1, 2, 0).numpy()

        # Save the image to images_list
        images_list.append(img)

        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()
        rects = ss.process()

        proposals = []
        features = []  # Add this line

        for i, rect in enumerate(rects):
            if i >= max_proposals_per_image:
                break
            x, y, width, height = rect 
            x, y, width, height = int(x), int(y), int(width), int(height)
            proposal_bbox = [x, y, width, height]  
            proposals.append(proposal_bbox)

            proposal_image = img[y:y+height, x:x+width]

            proposal_features = extract_vgg16_features(proposal_image) 
            features.append((proposal_bbox, np.ravel(proposal_features)))  # Save features together with bounding boxes

        proposals_list.append(proposals)
        features_list.extend(features)  # Add the features to the list

        labels = assign_labels(proposals, bboxes, image)
        data_list.extend(list(zip(proposals, labels)))

        counter += 1

    return proposals_list, data_list, features_list, images_list  # Return images_list

# def generate_proposals_and_labels(dataloader, ss, num_images_to_process, max_proposals_per_image):
#     proposals_list = []
#     data_list = []  # This will be a list of lists
#     images_list = []  # Add this line to keep the images
#     features_list = []  # This will be a list of lists

#     counter = 0

#     for image, bboxes in dataloader:
#         print(f"Processing image {counter + 1}/{num_images_to_process}")
#         if counter >= num_images_to_process:
#             break
#         img = image.squeeze().permute(1, 2, 0).numpy()

#         # Save the image to images_list
#         images_list.append(img)

#         ss.setBaseImage(img)
#         ss.switchToSelectiveSearchQuality()
#         rects = ss.process()

#         proposals = []
#         features = []  # This will store the features for this image
#         image_data = []  # This will store the data for this image

#         for i, rect in enumerate(rects):
#             if i >= max_proposals_per_image:
#                 break
#             x, y, width, height = rect 
#             x, y, width, height = int(x), int(y), int(width), int(height)
#             proposal_bbox = [x, y, width, height]  
#             proposals.append(proposal_bbox)

#             proposal_image = img[y:y+height, x:x+width]

#             proposal_features = extract_vgg16_features(proposal_image) 
#             features.append((proposal_bbox, np.ravel(proposal_features)))  # Save features together with bounding boxes

#             labels = assign_labels(proposals, bboxes, image)
#             image_data.append((proposal_bbox, labels))

#         proposals_list.append(proposals)
#         features_list.append(features)  # Add the features for this image to the main list
#         data_list.append(image_data)  # Add the data for this image to the main list

#         counter += 1

#     return proposals_list, data_list, features_list, images_list  # Return images_list




def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    hist = cv2.normalize(hist, hist).flatten()

    return hist

def extract_vgg16_features(image):
    image = cv2.resize(image, (224, 224))  # Resize to fit VGG16 input size
    image = torch.tensor(image).permute(2, 0, 1).float().to(device)  # Convert to torch tensor and move to GPU
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No need to compute gradients
        features = vgg16(image)

    features = features.cpu().numpy()  # Move features to CPU and convert to numpy array
    features = features.flatten()  # Flatten features to 1D array

    return features



def assign_labels(proposals, bboxes, image, iou_threshold=0.5):
    labels = []
    iou_scores = []

    image = image.squeeze().permute(1, 2, 0).numpy()

    for proposal in proposals:
        proposal_x1, proposal_y1, proposal_w, proposal_h = proposal
        proposal_area = proposal_w * proposal_h

        max_iou = 0.0  # Initialize max IOU score for each proposal

        for bbox in bboxes:
            bbox_x1, bbox_y1, bbox_w, bbox_h = bbox

            intersection_x1 = max(proposal_x1, bbox_x1)
            intersection_y1 = max(proposal_y1, bbox_y1)
            intersection_x2 = min(proposal_x1 + proposal_w, bbox_x1 + bbox_w)
            intersection_y2 = min(proposal_y1 + proposal_h, bbox_y1 + bbox_h)

            intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
            bbox_area = bbox_w * bbox_h
            union_area = proposal_area + bbox_area - intersection_area

            iou = intersection_area / union_area
            iou_scores.append(iou)

            if iou > max_iou:
                max_iou = iou

        if max_iou >= iou_threshold:
            labels.append(1)  # Object proposal is a positive example
        else:
            labels.append(0)  # Object proposal is a negative example

    num_ones = labels.count(1)
    print(f"Number of predictions: {num_ones}")

    # To visualize the bounding boxes and IOU scores, uncomment the following line
    #visualize_bboxes(image, proposals, bboxes, iou_scores)
    return labels




if __name__ == "__main__":

    DATA_PATH = "/dtu/datasets1/02514/data_wastedetection"
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    ctest = 0
    ctrain = 0

    dataset = WasteDatasetImages(transform=transforms.ToTensor(), resize=(224, 224))

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_images_to_process_train = 5#len(train_dataset)
    num_images_to_process_test = 5 #len(test_dataset)
    max_proposals_per_image = 1000

    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Loading VGG16 model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16 = models.vgg16(pretrained=True)
    vgg16 = torch.nn.Sequential(*(list(vgg16.children())[:-1]))
    vgg16.eval()
    vgg16 = vgg16.to(device)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    print("Generating proposals and labels for train set")
    train_proposals, train_data, train_features, _ = generate_proposals_and_labels(train_dataloader, ss, num_images_to_process_train, max_proposals_per_image)

    # Extract the features and their corresponding bounding boxes
    train_features_flat, train_boxes = zip(*[(feature, bbox) for bbox, feature in train_features])
    train_features_flat = list(train_features_flat)
    train_boxes = list(train_boxes)

    train_labels = [label for _, label in train_data]
    train_labels = np.array(train_labels)

    svm = LinearSVC(C=0.1, class_weight='balanced')
    svm.fit(train_features_flat, train_labels)
    print("Training done")
    print("Saving model")
    if not os.path.exists("models"):
        os.makedirs("models")

    dump(svm, 'models/svm_model.joblib')


    print("Generating proposals and labels for test set")
    test_proposals, test_data, test_features, test_images = generate_proposals_and_labels(test_dataloader, ss, num_images_to_process_test, max_proposals_per_image)

    # Extract the features and their corresponding bounding boxes
    test_features_flat, test_boxes = zip(*[(feature, bbox) for bbox, feature in test_features])
    test_features_flat = list(test_features_flat)
    test_boxes = list(test_boxes)

    test_labels = [label for _, label in test_data]
    test_labels = np.array(test_labels)

    svm = load('models/svm_model.joblib')

    # Make predictions on the test data
    predictions = svm.predict(test_features_flat)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    number_of_images = 5

    # predictions = []
    # for feature in test_features:
    #     bbox, feature_hist = feature 
    #     predictions.extend(svm.predict([feature_hist]))
    # for i, img in enumerate(test_images[:number_of_images]):
    #     print(f"[test_features[i][0]] {test_features[i][0]}")
    #     draw_bboxes(img, [test_features[i][0]], [predictions[i]], i)

   
    # A list to store the predictions for each image
    predictions_by_image = []

    # A list to store the bbox for each image
    bbox_list_by_image = []

    # You iterate over each feature in test_features
    print(f"test_features {test_features}")
    print(f"test_features[0] {test_features[0]}")
    print(f"len(test_features) {len(test_features)}")
    for feature in test_features:
        #print(f"feature {feature}")
        bbox, feature_hist = feature 
        prediction = svm.predict([feature_hist])[0]  # Assuming svm.predict() returns a list, taking the first element
        #print(f"prediction {prediction}")
        predictions_by_image.append(prediction)
        bbox_list_by_image.append(bbox)

    #print(f"predictions_by_image {predictions_by_image}")

    for i, img in enumerate(test_images[:number_of_images]):
        # Get the indices of the bounding boxes that the model predicts contain an object
        object_indices = [j for j, pred in enumerate(predictions_by_image) if pred == 1]
        
        # Get the corresponding bounding boxes
        object_bboxes = [bbox_list_by_image[index] for index in object_indices]

        # Draw the bounding boxes that the model predicts contain an object
        draw_bboxes(img, object_bboxes, [1] * len(object_bboxes), i)










