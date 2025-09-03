#!/usr/bin/env python3
"""
test_resize.py - test if resize works
"""

import cv2
import numpy as np

def test_resize():
    """test resize functionality"""
    
    # load original image
    img = cv2.imread('section_30_r01_c01.tif', cv2.IMREAD_UNCHANGED)
    if img is None:
        print("cannot load image")
        return
    
    print(f"original image size: {img.shape}")
    print(f"original image size: {img.nbytes / 1024 / 1024:.1f} MB")
    
    # apply 0.35 resize
    h, w = img.shape[:2]
    target_width = int(w * 0.35)
    target_height = int(h * 0.35)
    
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    
    print(f"resized image size: {resized.shape}")
    print(f"resized image size: {resized.nbytes / 1024 / 1024:.1f} MB")
    print(f"resize ratio: {target_width/w:.2f}x{target_height/h:.2f}")
    
    # save resized image
    cv2.imwrite('test_resize_only.tif', resized)
    print("saved to test_resize_only.tif")

if __name__ == "__main__":
    test_resize() 
