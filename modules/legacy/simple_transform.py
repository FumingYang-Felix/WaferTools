#!/usr/bin/env python3
"""
simple_transform.py - simple image transformation tool

features:
1. translation (dx, dy)
2. rotation (rotation)

usage:
    python simple_transform.py section_30_r01_c01.tif --dx -1500 --dy -4691 --rotation -0.94
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import os

def load_image(image_path):
    """load image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image file not found: {image_path}")
    
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {image_path}")
    
    print(f"loaded image: {image_path}")
    print(f"image size: {img.shape}")
    return img

def translate_image(img, dx, dy):
    """translate image"""
    if dx == 0 and dy == 0:
        return img
    
    h, w = img.shape[:2]
    
    # create translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # calculate new size
    new_width = w + abs(int(dx))
    new_height = h + abs(int(dy))
    
    # apply translation
    translated = cv2.warpAffine(
        img, 
        translation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"translation: dx={dx}, dy={dy}")
    print(f"translated size: {translated.shape}")
    return translated

def rotate_image(img, angle_deg, center=None):
    """rotate image"""
    if angle_deg == 0:
        return img
    
    h, w = img.shape[:2]
    
    # if center is not specified, use image center
    if center is None:
        center = (w // 2, h // 2)
    
    # create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # calculate rotated bounding box
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    # calculate new size
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    
    # adjust translation part of rotation matrix
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # apply rotation
    rotated = cv2.warpAffine(
        img, 
        rotation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"rotation: {angle_deg}°")
    print(f"rotated size: {rotated.shape}")
    return rotated

def save_image(img, output_path):
    """save image"""
    # ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # save image
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"image saved: {output_path}")
    else:
        print(f"save failed: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='simple image transformation tool')
    parser.add_argument('input_image', help='input image path')
    parser.add_argument('--dx', type=float, default=0, help='X direction translation pixel (default: 0)')
    parser.add_argument('--dy', type=float, default=0, help='Y direction translation pixel (default: 0)')
    parser.add_argument('--rotation', type=float, default=0, help='rotation angle (degree) (default: 0)')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'), 
                       help='rotation center point (default: image center)')
    parser.add_argument('--output', help='output image path (default: add _transform suffix)')
    parser.add_argument('--order', choices=['translate_first', 'rotate_first'], 
                       default='translate_first', help='transformation order (default: translate first)')
    
    args = parser.parse_args()
    
    try:
        # load image
        img = load_image(args.input_image)
        
        # apply transformation according to order
        if args.order == 'translate_first':
            # translate first, then rotate
            if args.dx != 0 or args.dy != 0:
                img = translate_image(img, args.dx, args.dy)
            if args.rotation != 0:
                img = rotate_image(img, args.rotation, args.center)
        else:
            # rotate first, then translate
            if args.rotation != 0:
                img = rotate_image(img, args.rotation, args.center)
            if args.dx != 0 or args.dy != 0:
                img = translate_image(img, args.dx, args.dy)
        
        # generate output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_image)
            output_path = str(input_path.parent / f"{input_path.stem}_transform{input_path.suffix}")
        
        # save image
        save_image(img, output_path)
        
        print("\ntransformation completed!")
        print(f"input: {args.input_image}")
        print(f"output: {output_path}")
        print(f"transformation parameters:")
        print(f"  Translation: dx={args.dx}, dy={args.dy}")
        print(f"  Rotation: {args.rotation}°")
        print(f"  transformation order: {args.order}")
        if args.center:
            print(f"  Center: ({args.center[0]}, {args.center[1]})")
        
    except Exception as e:
        print(f"error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
