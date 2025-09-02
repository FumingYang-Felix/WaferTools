#!/usr/bin/env python3
"""
direct_transform.py - 直接使用SIFT参数的图片变换工具

功能:
直接使用SIFT计算出的参数进行变换：
1. 平移 (dx, dy)
2. 旋转 (rotation) 
3. 缩放 (scale)

用法:
    python direct_transform.py section_30_r01_c01.tif --dx -1500.24 --dy -4691.47 --rotation -0.94 --scale 0.9985
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

def apply_scale(img, scale_x, scale_y):
    """应用缩放"""
    if scale_x == 1.0 and scale_y == 1.0:
        return img
    
    h, w = img.shape[:2]
    new_width = int(w * scale_x)
    new_height = int(h * scale_y)
    
    scaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"缩放: {scale_x:.4f}x{scale_y:.4f}")
    print(f"缩放后尺寸: {scaled.shape}")
    return scaled

def apply_translation(img, dx, dy):
    """应用平移"""
    if dx == 0 and dy == 0:
        return img
    
    h, w = img.shape[:2]
    
    # 创建平移矩阵
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # 计算新的尺寸
    new_width = w + abs(int(dx))
    new_height = h + abs(int(dy))
    
    # 应用平移
    translated = cv2.warpAffine(
        img, 
        translation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"平移: dx={dx:.2f}, dy={dy:.2f}")
    print(f"平移后尺寸: {translated.shape}")
    return translated

def apply_rotation(img, angle_deg, center=None):
    """应用旋转"""
    if angle_deg == 0:
        return img
    
    h, w = img.shape[:2]
    
    # 如果没有指定中心点，使用图片中心
    if center is None:
        center = (w // 2, h // 2)
    
    # 创建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # 计算旋转后的边界框
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    # 计算新的尺寸
    new_width = int((h * sin) + (w * cos))
    new_height = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵的平移部分
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 应用旋转
    rotated = cv2.warpAffine(
        img, 
        rotation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    print(f"旋转: {angle_deg:.4f}°")
    print(f"旋转后尺寸: {rotated.shape}")
    return rotated

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
    parser = argparse.ArgumentParser(description='直接使用SIFT参数的图片变换工具')
    parser.add_argument('input_image', help='输入图片路径')
    parser.add_argument('--dx', type=float, default=0, help='X方向平移像素 (默认: 0)')
    parser.add_argument('--dy', type=float, default=0, help='Y方向平移像素 (默认: 0)')
    parser.add_argument('--rotation', type=float, default=0, help='旋转角度(度) (默认: 0)')
    parser.add_argument('--scale', type=float, default=1.0, help='缩放比例 (默认: 1.0)')
    parser.add_argument('--center', nargs=2, type=float, metavar=('X', 'Y'), 
                       help='旋转中心点 (默认: 图片中心)')
    parser.add_argument('--output', help='输出图片路径 (默认: 添加_transform后缀)')
    parser.add_argument('--order', choices=['scale_first', 'translate_first', 'rotate_first'], 
                       default='scale_first', help='变换顺序 (默认: 先缩放)')
    
    args = parser.parse_args()
    
    try:
        # 加载图片
        img = load_image(args.input_image)
        
        # 根据顺序应用变换
        if args.order == 'scale_first':
            # 先缩放，后平移，最后旋转
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
        elif args.order == 'translate_first':
            # 先平移，后缩放，最后旋转
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
        else:  # rotate_first
            # 先旋转，后缩放，最后平移
            if args.rotation != 0:
                img = apply_rotation(img, args.rotation, args.center)
            if args.scale != 1.0:
                img = apply_scale(img, args.scale, args.scale)
            if args.dx != 0 or args.dy != 0:
                img = apply_translation(img, args.dx, args.dy)
        
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
        print(f"  Scale: {args.scale}")
        print(f"  Translation: dx={args.dx}, dy={args.dy}")
        print(f"  Rotation: {args.rotation}°")
        print(f"  变换顺序: {args.order}")
        if args.center:
            print(f"  Center: ({args.center[0]}, {args.center[1]})")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 