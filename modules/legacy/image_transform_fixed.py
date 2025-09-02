#!/usr/bin/env python3
"""
image_transform_fixed.py - 图片affine transformation工具 (基于原图)

功能:
进行affine transformation: scale -> translation -> rotation
(所有参数都基于原图尺寸)

用法:
    python image_transform_fixed.py section_30_r01_c01.tif --scale 1.2 --dx 50 --dy -30 --rotation 15
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import os

def load_image(image_path):
    """加载图片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    print(f"加载图片: {image_path}")
    print(f"图片尺寸: {img.shape}")
    return img

def create_affine_matrix(scale_x=1.0, scale_y=1.0, dx=0, dy=0, rotation_deg=0, center=None):
    """创建affine transformation矩阵"""
    # 如果没有指定中心点，使用图片中心
    if center is None:
        center = None  # 将在apply_affine_transform中计算
    
    # 创建变换矩阵
    # 1. Scale
    scale_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # 2. Translation
    translation_matrix = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    
    # 3. Rotation (degrees to radians)
    angle_rad = np.radians(rotation_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # 组合变换矩阵 (scale -> translation -> rotation)
    combined_matrix = rotation_matrix @ translation_matrix @ scale_matrix
    
    return combined_matrix

def apply_affine_transform(img, matrix, center=None):
    """应用affine transformation"""
    h, w = img.shape[:2]
    
    # 如果没有指定中心点，使用图片中心
    if center is None:
        center = (w // 2, h // 2)
    
    # 计算变换后的边界框
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ])
    
    # 应用变换
    transformed_corners = (matrix @ corners.T).T
    
    # 计算新的边界框
    min_x = int(np.floor(transformed_corners[:, 0].min()))
    max_x = int(np.ceil(transformed_corners[:, 0].max()))
    min_y = int(np.floor(transformed_corners[:, 1].min()))
    max_y = int(np.ceil(transformed_corners[:, 1].max()))
    
    # 计算偏移量
    offset_x = -min_x
    offset_y = -min_y
    
    # 创建新的变换矩阵，包含偏移
    final_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ]) @ matrix
    
    # 计算输出图片尺寸
    output_width = max_x - min_x
    output_height = max_y - min_y
    
    # 应用变换
    transformed = cv2.warpAffine(
        img, 
        final_matrix[:2, :], 
        (output_width, output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"变换后尺寸: {transformed.shape}")
    return transformed

def save_image(img, output_path):
    """保存图片"""
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图片
    success = cv2.imwrite(output_path, img)
    if success:
        print(f"图片已保存: {output_path}")
    else:
        print(f"保存失败: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='图片affine transformation工具 (基于原图)')
    parser.add_argument('input_image', help='输入图片路径')
    parser.add_argument('--scale', nargs=2, type=float, metavar=('SCALE_X', 'SCALE_Y'), 
                       default=[1.0, 1.0], help='缩放比例 (默认: 1.0 1.0)')
    parser.add_argument('--dx', type=float, default=0, help='X方向平移像素 (基于原图尺寸, 默认: 0)')
    parser.add_argument('--dy', type=float, default=0, help='Y方向平移像素 (基于原图尺寸, 默认: 0)')
    parser.add_argument('--rotation', type=float, default=0, help='旋转角度(度) (默认: 0)')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'), 
                       help='旋转中心点 (默认: 图片中心)')
    parser.add_argument('--output', help='输出图片路径 (默认: 添加_transform后缀)')
    
    args = parser.parse_args()
    
    try:
        # 加载图片
        img = load_image(args.input_image)
        
        # 创建affine transformation矩阵
        matrix = create_affine_matrix(
            scale_x=args.scale[0],
            scale_y=args.scale[1],
            dx=args.dx,
            dy=args.dy,
            rotation_deg=args.rotation,
            center=tuple(args.center) if args.center else None
        )
        
        # 应用变换
        if any([args.scale != [1.0, 1.0], args.dx != 0, args.dy != 0, args.rotation != 0]):
            img = apply_affine_transform(img, matrix, args.center)
        
        # 生成输出路径
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input_image)
            output_path = str(input_path.parent / f"{input_path.stem}_transform{input_path.suffix}")
        
        # 保存图片
        save_image(img, output_path)
        
        print("\n变换完成!")
        print(f"输入: {args.input_image}")
        print(f"输出: {output_path}")
        print(f"变换参数:")
        print(f"  Scale: {args.scale[0]}x{args.scale[1]}")
        print(f"  Translation: dx={args.dx}, dy={args.dy} (基于原图尺寸)")
        print(f"  Rotation: {args.rotation}°")
        if args.center:
            print(f"  Center: ({args.center[0]}, {args.center[1]})")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 