
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./GradCamResults/heatmap.pkl', 'rb') as f:
    heatmap = pickle.load(f)

heatmap = heatmap.numpy()
stdev = 0.15 * (np.max(heatmap) - np.min(heatmap))

noise = np.random.normal(0, stdev, heatmap.shape).astype(np.float32)
x_plus_noise = heatmap + noise

x_plus_noise = np.clip(x_plus_noise, 0, 255).astype(np.uint8)

plt.imshow(x_plus_noise)
plt.savefig('./GradCamResults/heatmap_noise.jpg')




image = cv2.imread('./GradCamResults/image.jpg')

print(heatmap.shape)
print(image.shape)

heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
plt.imshow(heatmap)
plt.savefig('./GradCamResults/heatmap_smooth.jpg')
superimposed_img = heatmap * 0.4 + image
cv2.imwrite('./GradCamResults/map_noise.jpg', superimposed_img)