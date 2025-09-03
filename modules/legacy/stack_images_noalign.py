#!/usr/bin/env python3
"""
Stack images by order, no alignment, output as multi-page TIFF.
"""

import os
import cv2
import numpy as np
import tifffile
from tqdm import tqdm

ORDER_FILE = 'improved_order.txt'
IMAGE_DIR = 'w7_png_4k'
OUTPUT_STACK = 'stacked_noalign_4k.tif'

# read order
with open(ORDER_FILE, 'r') as f:
    order = f.read().strip().split()

stack = []
for section in tqdm(order, desc='Stacking'):
    img_path = os.path.join(IMAGE_DIR, f'{section}.png')
    if not os.path.exists(img_path):
        print(f'Warning: {img_path} not found, skipping.')
        continue
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Warning: {img_path} could not be loaded, skipping.')
        continue
    stack.append(img)

if len(stack) == 0:
    print('No images loaded!')
    exit(1)

stack = np.stack(stack, axis=0)
print(f'Saving stack: {OUTPUT_STACK}, shape={stack.shape}')
tifffile.imwrite(OUTPUT_STACK, stack)
print('Done.') 
