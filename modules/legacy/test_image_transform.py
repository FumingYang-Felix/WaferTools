#!/usr/bin/env python3
"""
test_image_transform.py - 测试image_transform.py的使用方法
"""

import subprocess
import sys
import os

def run_transform_test():
    """运行transform测试"""
    
    # 检查是否有测试图片
    test_image = "section_30_r01_c01.tif"
    if not os.path.exists(test_image):
        print(f"测试图片不存在: {test_image}")
        print("请确保有测试图片文件，或者修改脚本中的图片路径")
        return
    
    print("=== 图片Transform工具测试 ===\n")
    
    # 测试1: 只resize
    print("测试1: 只resize (0.5x0.5)")
    cmd1 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.5", "0.5",
        "--output", "test_resize_only.tif"
    ]
    subprocess.run(cmd1)
    print()
    
    # 测试2: 只scale
    print("测试2: 只scale (1.2x1.2)")
    cmd2 = [
        sys.executable, "image_transform.py", test_image,
        "--scale", "1.2", "1.2",
        "--output", "test_scale_only.tif"
    ]
    subprocess.run(cmd2)
    print()
    
    # 测试3: 只translation
    print("测试3: 只translation (dx=50, dy=-30)")
    cmd3 = [
        sys.executable, "image_transform.py", test_image,
        "--dx", "50", "--dy", "-30",
        "--output", "test_translation_only.tif"
    ]
    subprocess.run(cmd3)
    print()
    
    # 测试4: 只rotation
    print("测试4: 只rotation (15度)")
    cmd4 = [
        sys.executable, "image_transform.py", test_image,
        "--rotation", "15",
        "--output", "test_rotation_only.tif"
    ]
    subprocess.run(cmd4)
    print()
    
    # 测试5: 组合变换
    print("测试5: 组合变换 (resize + scale + translation + rotation)")
    cmd5 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.8", "0.8",
        "--scale", "1.1", "1.1",
        "--dx", "30", "--dy", "-20",
        "--rotation", "10",
        "--output", "test_combined.tif"
    ]
    subprocess.run(cmd5)
    print()
    
    print("=== 测试完成 ===")
    print("生成的文件:")
    print("  - test_resize_only.tif")
    print("  - test_scale_only.tif")
    print("  - test_translation_only.tif")
    print("  - test_rotation_only.tif")
    print("  - test_combined.tif")

def show_usage_examples():
    """显示使用示例"""
    print("=== 使用示例 ===\n")
    
    examples = [
        {
            "description": "只resize到0.5倍",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.5 0.5"
        },
        {
            "description": "只缩放1.2倍",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.2 1.2"
        },
        {
            "description": "只平移(dx=50, dy=-30)",
            "command": "python image_transform.py section_30_r01_c01.tif --dx 50 --dy -30"
        },
        {
            "description": "只旋转15度",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 15"
        },
        {
            "description": "组合变换",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.8 0.8 --scale 1.2 --dx 50 --dy -30 --rotation 15"
        },
        {
            "description": "指定旋转中心点",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 45 --center 512 512"
        },
        {
            "description": "指定输出文件名",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.5 --output my_transformed_image.tif"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

if __name__ == "__main__":
    print("图片Transform工具演示\n")
    
    # 显示使用示例
    show_usage_examples()
    
    # 询问是否运行测试
    response = input("是否运行测试? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_transform_test()
    else:
        print("跳过测试。") 
"""
test_image_transform.py - 测试image_transform.py的使用方法
"""

import subprocess
import sys
import os

def run_transform_test():
    """运行transform测试"""
    
    # 检查是否有测试图片
    test_image = "section_30_r01_c01.tif"
    if not os.path.exists(test_image):
        print(f"测试图片不存在: {test_image}")
        print("请确保有测试图片文件，或者修改脚本中的图片路径")
        return
    
    print("=== 图片Transform工具测试 ===\n")
    
    # 测试1: 只resize
    print("测试1: 只resize (0.5x0.5)")
    cmd1 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.5", "0.5",
        "--output", "test_resize_only.tif"
    ]
    subprocess.run(cmd1)
    print()
    
    # 测试2: 只scale
    print("测试2: 只scale (1.2x1.2)")
    cmd2 = [
        sys.executable, "image_transform.py", test_image,
        "--scale", "1.2", "1.2",
        "--output", "test_scale_only.tif"
    ]
    subprocess.run(cmd2)
    print()
    
    # 测试3: 只translation
    print("测试3: 只translation (dx=50, dy=-30)")
    cmd3 = [
        sys.executable, "image_transform.py", test_image,
        "--dx", "50", "--dy", "-30",
        "--output", "test_translation_only.tif"
    ]
    subprocess.run(cmd3)
    print()
    
    # 测试4: 只rotation
    print("测试4: 只rotation (15度)")
    cmd4 = [
        sys.executable, "image_transform.py", test_image,
        "--rotation", "15",
        "--output", "test_rotation_only.tif"
    ]
    subprocess.run(cmd4)
    print()
    
    # 测试5: 组合变换
    print("测试5: 组合变换 (resize + scale + translation + rotation)")
    cmd5 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.8", "0.8",
        "--scale", "1.1", "1.1",
        "--dx", "30", "--dy", "-20",
        "--rotation", "10",
        "--output", "test_combined.tif"
    ]
    subprocess.run(cmd5)
    print()
    
    print("=== 测试完成 ===")
    print("生成的文件:")
    print("  - test_resize_only.tif")
    print("  - test_scale_only.tif")
    print("  - test_translation_only.tif")
    print("  - test_rotation_only.tif")
    print("  - test_combined.tif")

def show_usage_examples():
    """显示使用示例"""
    print("=== 使用示例 ===\n")
    
    examples = [
        {
            "description": "只resize到0.5倍",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.5 0.5"
        },
        {
            "description": "只缩放1.2倍",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.2 1.2"
        },
        {
            "description": "只平移(dx=50, dy=-30)",
            "command": "python image_transform.py section_30_r01_c01.tif --dx 50 --dy -30"
        },
        {
            "description": "只旋转15度",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 15"
        },
        {
            "description": "组合变换",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.8 0.8 --scale 1.2 --dx 50 --dy -30 --rotation 15"
        },
        {
            "description": "指定旋转中心点",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 45 --center 512 512"
        },
        {
            "description": "指定输出文件名",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.5 --output my_transformed_image.tif"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

if __name__ == "__main__":
    print("图片Transform工具演示\n")
    
    # 显示使用示例
    show_usage_examples()
    
    # 询问是否运行测试
    response = input("是否运行测试? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_transform_test()
    else:
        print("跳过测试。") 
 
 
 
 
 
 