from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import re

basic_transform = transforms.Compose([
        transforms.Resize((224,224)), # resizing to same size, for example 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) # normalization settings for RGB

def get_dataLoader(data_aug = False, batch_size = 64):

    current_path = os.getcwd()
    train_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'train')
    test_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'test')
    
    if data_aug == True:
        transform = transforms.Compose([
        transforms.Resize((224,224)), # resizing to same size, for example 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization settings for RGB
    ])
    else: 
        transform = basic_transform

    trainset = datasets.ImageFolder(train_path, transform=transform)
    testset = datasets.ImageFolder(test_path, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)


    return trainset, testset, train_loader, test_loader

def parse_log_file(log_file):
    with open(log_file, 'r') as f:
        log_lines = f.readlines()

    epochs = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    pattern = re.compile(r'Epoch: (\d+), Loss train: ([\d\.]+), Loss test: ([\d\.]+), Accuracy train: ([\d\.]+)%, Accuracy test: ([\d\.]+)%')

    for line in log_lines:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            test_losses.append(float(match.group(3)))
            train_accuracies.append(float(match.group(4)))
            test_accuracies.append(float(match.group(5)))

    return epochs, train_losses, test_losses, train_accuracies, test_accuracies

def plot_metrics(epochs, train_losses, test_losses, train_accuracies, test_accuracies):
    # Plot loss
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Loss_resnet18_pretrained.png') # Save figure
    plt.close() # Close figure

    # Plot accuracy
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accuracies, label='Train accuracy')
    plt.plot(epochs, test_accuracies, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/Accuracy_resnet18_pretrained.png') # Save figure
    plt.close() # Close figure





if __name__ == "__main__":

    current_path = os.getcwd()
    log_file = os.path.join(current_path, 'logs', 'resnet18_pretrained.log')
    epochs, train_losses, test_losses, train_accuracies, test_accuracies = parse_log_file(log_file)
    plot_metrics(epochs, train_losses, test_losses, train_accuracies, test_accuracies)