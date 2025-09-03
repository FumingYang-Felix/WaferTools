import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import cv2

img = cv2.cvtColor(cv2.imread('1750864145001.jpg'), cv2.COLOR_BGR2RGB)

# 1) RGB -> Lab
lab = color.rgb2lab(img)

# 2) calculate chroma and optional blur
chroma = np.hypot(lab[...,1], lab[...,2])
chroma_blur = cv2.GaussianBlur(chroma,(0,0),sigmaX=3)

# 3) visualize
plt.figure(figsize=(6,6))
plt.imshow(chroma_blur, cmap='inferno')  # also 'viridis', 'jet', ...
plt.colorbar(label='Lab chroma')
plt.axis('off')
plt.tight_layout()
plt.savefig('chroma_heatmap.png', dpi=300)
plt.show()
