#!/usr/bin/env python3
"""
测试自动mask生成器功能
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from section_counter import process_image_with_auto_generator, process_image_with_sam

def test_auto_generator():
    """测试自动mask生成器功能"""
    
    # 检查是否有测试图像
    test_images = [
        "1750864145001.jpg",  # 你目录中的图像
        "w06_250nm_1.0mms_60sections_RGB.png_files.png"  # 当前项目的图像
    ]
    
    for img_name in test_images:
        if os.path.exists(img_name):
            print(f"🔍 测试图像: {img_name}")
            
            # 测试自动mask生成器
            print("📊 测试自动mask生成器模式...")
            try:
                result_auto = process_image_with_auto_generator(img_name, 'vit_l')
                if result_auto:
                    print("✅ 自动mask生成器测试成功")
                else:
                    print("❌ 自动mask生成器测试失败")
            except Exception as e:
                print(f"❌ 自动mask生成器错误: {e}")
            
            # 测试原始网格点模式
            print("📊 测试改进的网格点模式...")
            try:
                result_grid = process_image_with_sam(img_name, 'vit_l', use_auto_generator=False)
                if result_grid:
                    print("✅ 改进网格点模式测试成功")
                else:
                    print("❌ 改进网格点模式测试失败")
            except Exception as e:
                print(f"❌ 改进网格点模式错误: {e}")
            
            break
    else:
        print("❌ 未找到测试图像")
        print("请确保有以下图像之一:")
        for img in test_images:
            print(f"  - {img}")

if __name__ == "__main__":
    test_auto_generator() 