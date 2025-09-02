#!/usr/bin/env python3
"""
test_resize.py - 测试resize是否生效
"""

import cv2
import numpy as np

def test_resize():
    """测试resize功能"""
    
    # 加载原始图片
    img = cv2.imread('section_30_r01_c01.tif', cv2.IMREAD_UNCHANGED)
    if img is None:
        print("无法加载图片")
        return
    
    print(f"原始图片尺寸: {img.shape}")
    print(f"原始图片大小: {img.nbytes / 1024 / 1024:.1f} MB")
    
    # 应用0.35的resize
    h, w = img.shape[:2]
    target_width = int(w * 0.35)
    target_height = int(h * 0.35)
    
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"Resize后尺寸: {resized.shape}")
    print(f"Resize后大小: {resized.nbytes / 1024 / 1024:.1f} MB")
    print(f"Resize比例: {target_width/w:.2f}x{target_height/h:.2f}")
    
    # 保存resize后的图片
    cv2.imwrite('test_resize_only.tif', resized)
    print("已保存: test_resize_only.tif")

if __name__ == "__main__":
    test_resize() 