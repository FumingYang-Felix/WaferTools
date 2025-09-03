#!/usr/bin/env python3
"""
direct_transform.py - direct image transformation tool using SIFT parameters

function:
directly use SIFT calculated parameters for transformation:
1. translation (dx, dy)
2. rotation (rotation) 
3. scaling (scale)

usage:
    python direct_transform.py section_30_r01_c01.tif --dx -1500.24 --dy -4691.47 --rotation -0.94 --scale 0.9985
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
    
    print(f"load image: {image_path}")
    print(f"image size: {img.shape}")
    return img

def apply_scale(img, scale_x, scale_y):
    """apply scaling"""
    if scale_x == 1.0 and scale_y == 1.0:
        return img
    
    h, w = img.shape[:2]
    new_width = int(w * scale_x)
    new_height = int(h * scale_y)
    
    scaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"scaling: {scale_x:.4f}x{scale_y:.4f}")
    print(f"scaled size: {scaled.shape}")
    return scaled

def apply_translation(img, dx, dy):
    """apply translation"""
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
    
    print(f"translation: dx={dx:.2f}, dy={dy:.2f}")
    print(f"translated size: {translated.shape}")
    return translated

def apply_rotation(img, angle_deg, center=None):
    """apply rotation"""
    if angle_deg == 0:
        return img
    
    h, w = img.shape[:2]
    
    # if center is not specified, use image center
    if center is None:
        center = (w // 2, h // 2)
    
    # create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # calculate new size
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
    
    print(f"rotation: {angle_deg:.4f}°")
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
    parser = argparse.ArgumentParser(description='direct image transformation tool using SIFT parameters')
    parser.add_argument('input_image', help='input image path')
    parser.add_argument('--dx', type=float, default=0, help='X direction translation pixel (default: 0)')
    parser.add_argument('--dy', type=float, default=0, help='Y direction translation pixel (default: 0)')
    parser.add_argument('--rotation', type=float, default=0, help='rotation angle (degree) (default: 0)')
    parser.add_argument('--scale', type=float, default=1.0, help='scaling ratio (default: 1.0)')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'), 
                       help='rotation center point (default: image center)')
    parser.add_argument('--output', help='output image path (default: add _transform suffix)')
    parser.add_argument('--order', choices=['scale_first', 'translate_first', 'rotate_first'], 
                       default='scale_first', help='transformation order (default: scale first)')
    
    args = parser.parse_args()
    
    try:
        # load image
        img = load_image(args.input_image)
        
        # apply transformation according to order
        if args.order == 'scale_first':
            # scale first, then translation, then rotation
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
        elif args.order == 'translate_first':
            # translate first, then scale, then rotation
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
        else:  # rotate_first
            # rotate first, then scale, then translation
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
        
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
        print(f"  Scale: {args.scale}")
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
