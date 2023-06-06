import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from model.model_P import Network
from data_utils import get_dataLoader

#Setup Device
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infere(model, img):
    img = img.to(device)
    with torch.no_grad():
        out = model(img)
    prediction = out.argmax(1)
    return prediction

def check_problematic_images(model):
    _, test_set, _, _ = get_dataLoader()

    incorrect_images = []
    for img, target in test_set:
        data = torch.tensor(np.array(img))
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        predicted = output.argmax(1)
        if target != predicted: incorrect_images.append(img.filename)

    return incorrect_images


def test(model):
    _, _, _, test_loader = get_dataLoader()

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
        test_outs.append(soft_out.cpu().numpy())
        test_correct += (target==predicted).sum().cpu().item()
        test_labels.extend(target.cpu().numpy())
        test_preds.extend(predicted.cpu().numpy())
        
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_acc = test_correct/len(test_loader)
    test_auc = roc_auc_score(test_labels, test_outs)

    return test_acc, test_f1, test_auc

if __name__ == '__main__':
    args = sys.argv[1:] if len(sys.argv) > 1 else None
    if len(args) == 1:
        model_name = args[1]
        model = Network().to(device).load_state_dict(f'trained_models/{model_name}.pth')
        test(model)

    if len(args) == 2:
        model_name, img_path = args
        model = Network().to(device).load_state_dict(f'trained_models/{model_name}.pth')
        img = Image.open(img_path)
        img_tensor = torch.tensor(np.array(img))
        print(infere(model, img))
