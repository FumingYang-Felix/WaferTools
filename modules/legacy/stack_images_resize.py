#!/usr/bin/env python3
import os
import tifffile
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import argparse


def stack_images_with_resize(order_file, image_dir, output_file, size=2048):
    """
    按顺序读取图片，resize为size*size，堆叠保存为tif
    """
    # 读取顺序
    with open(order_file) as f:
        order = f.read().strip().split()
    print(f"Loaded order with {len(order)} sections")

    stack = []
    for idx, section in enumerate(order):
        img_path = os.path.join(image_dir, f"{section}.png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        img = imread(img_path)
        if img.shape[0] != size or img.shape[1] != size:
            img_resized = resize(img, (size, size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            img_resized = img
        stack.append(img_resized)
        if (idx+1) % 10 == 0 or idx == 0:
            print(f"Loaded and resized {idx+1}/{len(order)}: {section}")
    stack = np.stack(stack, axis=0)
    print(f"Stack shape: {stack.shape}")
    tifffile.imwrite(output_file, stack)
    print(f"Saved stack to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack images with resize")
    parser.add_argument("order_file", help="Order txt file")
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("--output", help="Output tif file")
    parser.add_argument("--size", type=int, default=2048, help="Resize to size x size")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"stack_{args.size}.tif"
    stack_images_with_resize(args.order_file, args.image_dir, args.output, args.size) 
import os
import tifffile
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import argparse


def stack_images_with_resize(order_file, image_dir, output_file, size=2048):
    """
    按顺序读取图片，resize为size*size，堆叠保存为tif
    """
    # 读取顺序
    with open(order_file) as f:
        order = f.read().strip().split()
    print(f"Loaded order with {len(order)} sections")

    stack = []
    for idx, section in enumerate(order):
        img_path = os.path.join(image_dir, f"{section}.png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue
        img = imread(img_path)
        if img.shape[0] != size or img.shape[1] != size:
            img_resized = resize(img, (size, size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        else:
            img_resized = img
        stack.append(img_resized)
        if (idx+1) % 10 == 0 or idx == 0:
            print(f"Loaded and resized {idx+1}/{len(order)}: {section}")
    stack = np.stack(stack, axis=0)
    print(f"Stack shape: {stack.shape}")
    tifffile.imwrite(output_file, stack)
    print(f"Saved stack to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack images with resize")
    parser.add_argument("order_file", help="Order txt file")
    parser.add_argument("image_dir", help="Image directory")
    parser.add_argument("--output", help="Output tif file")
    parser.add_argument("--size", type=int, default=2048, help="Resize to size x size")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"stack_{args.size}.tif"
    stack_images_with_resize(args.order_file, args.image_dir, args.output, args.size) 
 
 
 
 
 
 