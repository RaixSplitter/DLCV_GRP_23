import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from model.model_P import Network
from data_utils import get_dataLoader, basic_transform

CLASSES = ['hotdog', 'nothotdog']

#Setup Device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infere(model, img):
    img_tensor = basic_transform(img)
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        #unsqueeze makes a batch of one image
        out = model(img_tensor.unsqueeze(0))
    prediction = out.argmax(1)
    return prediction.cpu().item()

#not done because dataset returns a tensor and not the image, so the filename cannot be read... :(
def check_problematic_images(model):
    current_path = os.getcwd()
    test_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'test')
    
    incorrect_images = []
    for path, dirs, files in os.walk(test_path):
        if files and not dirs:
            target = 0 if 'hotdog' == path.split('/')[-1] else 1
            for img_name in files:
                img = Image.open(os.path.join(path,img_name))
                data = basic_transform(img)
                data = data.to(device)
                model.eval()
                with torch.no_grad():
                    output = model(data.unsqueeze(0))
                predicted = output.argmax(1)
                if target != predicted.cpu().item():
                    incorrect_images.append(f'{img_name} pred {predicted.cpu().item()}')

    return incorrect_images


def test(model):
    _, _, test_set, test_loader = get_dataLoader()

    def loss_fun(output, target):
        target_one_hot = F.one_hot(target, num_classes=2).float()
        return F.binary_cross_entropy_with_logits(output, target_one_hot)

    test_loss = []
    test_correct = 0
    test_labels, test_preds, test_outs = [], [], []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss.append(loss_fun(output, target).cpu().item())
        predicted = output.argmax(1)
        soft_out = nn.Softmax(dim=1)(output)
        test_outs.extend(soft_out.cpu().numpy())
        test_correct += (target==predicted).sum().cpu().item()
        test_labels.extend(target.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())
        
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_acc = test_correct/len(test_loader)
    test_auc = roc_auc_score(test_labels, np.array(test_outs)[:,1])

    return test_acc, test_f1, test_auc

if __name__ == '__main__':
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    if args is not None:
        if len(args) == 2:
            model_name = args[1]
            model = Network()
            model.load_state_dict(torch.load(f'trained_models/{model_name}.pth'))
            model.to(device)
            if args[0] == 'test':
                acc, f1, auc = test(model)
                print(f'Model {model_name} scores: Accuracy: {acc}, F1: {f1}, ROC AUC: {auc}')

            elif args[0] == 'misspredictions':
                incorrect_preds = check_problematic_images(model)
                print(f'Model {model_name} incorrect predictions: {incorrect_preds}')

        if len(args) == 3:
            if args[0] == 'infere':
                model_name, img_path = args[1:]
                model = Network()
                model.to(device)
                model.load_state_dict(torch.load(f'trained_models/{model_name}.pth'))
                img = Image.open(img_path)
                pred = infere(model, img)
                print(f'Predicted class: {pred} ({CLASSES[int(pred)]})')
