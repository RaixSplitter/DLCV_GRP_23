from typing import Any, Mapping
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torchvision.models as models
import pickle


# Our imports
from data_utils import get_dataLoader
from model.model_gradmod import Network_Grad_Mod as Network
from model.gradcam_models import Network_Grad_Mod, Resnet_Grad_Mod





if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataLoader():

    current_path = os.getcwd()
    train_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'train')
    test_path = os.path.join(current_path, '..', '..', '02514', 'hotdog_nothotdog', 'test')
    

    transform = transforms.Compose([
    transforms.Resize((224,224)), # resizing to same size, for example 224x224
    transforms.ToTensor(),])


    trainset = datasets.ImageFolder(train_path, transform=transform)
    testset = datasets.ImageFolder(test_path, transform=transform)

    attempt = 0
    while attempt <= 10:
        #Get a random image from the test set and its path
        random_image_index = np.random.randint(0, len(testset))
        random_image = testset[random_image_index][0]
        random_image_path = testset.imgs[random_image_index][0]
        random_image = random_image.reshape((1, 3, 224, 224))

        #Check which class the image belongs to
        if random_image_path.split('/')[-2] == 'hotdog':
            print(random_image_path)
            break
        
        attempt += 1
    return random_image, random_image_path

img, img_path = get_dataLoader()





# save_model_path = f"trained_models/NN.pth"
# model = Network_Grad_Mod(save_model_path)
save_model_path = f"trained_models/resnet.pth"
model = Resnet_Grad_Mod(save_model_path)


model.eval()
    
pred = model(img)

print(pred)
print(pred.argmax(dim=1))


pred[:, 1].backward()
# pred.backward()
gradients = model.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
activations = model.get_activations(img).detach()


for i in range(32):
    activations[:, i, :, :] *= pooled_gradients[i]


heatmap = torch.mean(activations, dim=1).squeeze()
print(heatmap.max(), heatmap.min())

heatmap = np.maximum(heatmap, 0)
print(heatmap.max(), heatmap.min())

heatmap /= torch.max(heatmap)
print(heatmap)

# save heatmap as pickle
with open('./GradCamResults/heatmap.pkl', 'wb') as f:
    pickle.dump(heatmap, f)


print(heatmap.max(), heatmap.min())
image = cv2.imread(img_path)

cv2.imwrite('./GradCamResults/image.jpg', image)
plt.matshow(heatmap.squeeze())
plt.savefig('./GradCamResults/heatmap.jpg')




heatmap = cv2.resize(heatmap.numpy(), (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + image
cv2.imwrite('./GradCamResults/map.jpg', superimposed_img)


