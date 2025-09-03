#!/usr/bin/env python3
"""
image_transform_fixed.py - image affine transformation tool (based on original image)

features:
1. affine transformation: scale -> translation -> rotation
   (all parameters are based on original image size)

usage:
    python image_transform_fixed.py section_30_r01_c01.tif --scale 1.2 --dx 50 --dy -30 --rotation 15
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

def create_affine_matrix(scale_x=1.0, scale_y=1.0, dx=0, dy=0, rotation_deg=0, center=None):
    """create affine transformation matrix"""
    # if center is not specified, use image center
    if center is None:
        center = None  # will be calculated in apply_affine_transform
    
    # create transformation matrix
    # 1. Scale
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # 2. Translation (translation parameters will be adjusted automatically based on resize ratio)
    translation_matrix = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    
    # 3. Rotation (convert degrees to radians)
    angle_rad = np.radians(rotation_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # combine transformation matrices (scale -> translation -> rotation)
    combined_matrix = rotation_matrix @ translation_matrix @ scale_matrix
    
    return combined_matrix

def apply_affine_transform(img, matrix, center=None):
    """apply affine transformation"""
    h, w = img.shape[:2]
    
    # if center is not specified, use image center
    if center is None:
        center = (w // 2, h // 2)
    
    # calculate transformed bounding box
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ])
    
    # apply transformation
    transformed_corners = (matrix @ corners.T).T
    
    # calculate new bounding box
    min_x = int(np.floor(transformed_corners[:, 0].min()))
    max_x = int(np.ceil(transformed_corners[:, 0].max()))
    min_y = int(np.floor(transformed_corners[:, 1].min()))
    max_y = int(np.ceil(transformed_corners[:, 1].max()))
    
    # calculate offset
    offset_x = -min_x
    offset_y = -min_y
    
    # create new transformation matrix, including offset
    final_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ]) @ matrix
    
    # calculate output image size
    output_width = max_x - min_x
    output_height = max_y - min_y
    
    # apply transformation
    transformed = cv2.warpAffine(
        img, 
        final_matrix[:2, :], 
        (output_width, output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"transformed size: {transformed.shape}")
    return transformed

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
    parser = argparse.ArgumentParser(description='image affine transformation tool (based on original image)')
    parser.add_argument('input_image', help='input image path')
    parser.add_argument('--scale', nargs=2, type=float, metavar=('SCALE_X', 'SCALE_Y'), 
                       default=[1.0, 1.0], help='scale ratio (default: 1.0 1.0)')
    parser.add_argument('--dx', type=float, default=0, help='X direction translation pixel (based on original image size, default: 0)')
    parser.add_argument('--dy', type=float, default=0, help='Y direction translation pixel (based on original image size, default: 0)')
    parser.add_argument('--rotation', type=float, default=0, help='rotation angle(degree) (default: 0)')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'), 
                       help='rotation center point (default: image center)')
    parser.add_argument('--output', help='output image path (default: add _transform suffix)')
    
    args = parser.parse_args()
    
    try:
        # load image
        img = load_image(args.input_image)
        
        # create affine transformation matrix
        matrix = create_affine_matrix(
            scale_x=args.scale[0],
            scale_y=args.scale[1],
            dx=args.dx,
            dy=args.dy,
            rotation_deg=args.rotation,
            center=tuple(args.center) if args.center else None
        )
        
        # apply transformation
        if any([args.scale != [1.0, 1.0], args.dx != 0, args.dy != 0, args.rotation != 0]):
            img = apply_affine_transform(img, matrix, args.center)
        
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
        print(f"  Scale: {args.scale[0]}x{args.scale[1]}")
        print(f"  Translation: dx={args.dx}, dy={args.dy} (based on original image size)")
        print(f"  Rotation: {args.rotation}°")
        if args.center:
            print(f"  Center: ({args.center[0]}, {args.center[1]})")
        
    except Exception as e:
        print(f"error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
