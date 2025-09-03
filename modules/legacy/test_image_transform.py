#!/usr/bin/env python3
"""
test_image_transform.py - test image_transform.py
"""

import subprocess
import sys
import os

def run_transform_test():
    """run transform test"""
    
    # check if there are test images
    test_image = "section_30_r01_c01.tif"
    if not os.path.exists(test_image):
        print(f"test image not found: {test_image}")
        print("please ensure there is a test image file, or modify the image path in the script")
        return
    
    print("=== image transform tool test ===\n")
    
    # test 1: only resize
    print("test 1: only resize (0.5x0.5)")
    cmd1 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.5", "0.5",
        "--output", "test_resize_only.tif"
    ]
    subprocess.run(cmd1)
    print()
    
    # test 2: only scale
    print("test 2: only scale (1.2x1.2)")
    cmd2 = [
        sys.executable, "image_transform.py", test_image,
        "--scale", "1.2", "1.2",
        "--output", "test_scale_only.tif"
    ]
    subprocess.run(cmd2)
    print()
    
    # test 3: only translation
    print("test 3: only translation (dx=50, dy=-30)")
    cmd3 = [
        sys.executable, "image_transform.py", test_image,
        "--dx", "50", "--dy", "-30",
        "--output", "test_translation_only.tif"
    ]
    subprocess.run(cmd3)
    print()
    
    # test 4: only rotation
    print("test 4: only rotation (15 degrees)")
    cmd4 = [
        sys.executable, "image_transform.py", test_image,
        "--rotation", "15",
        "--output", "test_rotation_only.tif"
    ]
    subprocess.run(cmd4)
    print()
    
    # test 5: combined transformation
    print("test 5: combined transformation (resize + scale + translation + rotation)")
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
    
    print("=== test completed ===")
    print("generated files:")
    print("  - test_resize_only.tif")
    print("  - test_scale_only.tif")
    print("  - test_translation_only.tif")
    print("  - test_rotation_only.tif")
    print("  - test_combined.tif")

def show_usage_examples():
    """show usage examples"""
    print("=== usage examples ===\n")
    
    examples = [
        {
            "description": "only resize to 0.5x",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.5 0.5"
        },
        {
            "description": "only scale to 1.2x",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.2 1.2"
        },
        {
            "description": "only translation (dx=50, dy=-30)",
            "command": "python image_transform.py section_30_r01_c01.tif --dx 50 --dy -30"
        },
        {
            "description": "only rotation (15 degrees)",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 15"
        },
        {
            "description": "combined transformation",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.8 0.8 --scale 1.2 --dx 50 --dy -30 --rotation 15"
        },
        {
            "description": "specify rotation center point",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 45 --center 512 512"
        },
        {
            "description": "specify output file name",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.5 --output my_transformed_image.tif"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

if __name__ == "__main__":
    print("image transform tool demo\n")
    
    # show usage examples
    show_usage_examples()
    
    # ask if run test
    response = input("run test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_transform_test()
    else:
        print("skip test.") 
"""
test_image_transform.py - test image_transform.py
"""

import subprocess
import sys
import os

def run_transform_test():
    """run transform test"""
    
    # check if there are test images
    test_image = "section_30_r01_c01.tif"
    if not os.path.exists(test_image):
        print(f"test image not found: {test_image}")
        print("please ensure there is a test image file, or modify the image path in the script")
        return
    
    print("=== image transform tool test ===\n")
    
    # test 1: only resize
    print("test 1: only resize (0.5x0.5)")
    cmd1 = [
        sys.executable, "image_transform.py", test_image,
        "--resize", "0.5", "0.5",
        "--output", "test_resize_only.tif"
    ]
    subprocess.run(cmd1)
    print()
    
    # test 2: only scale
    print("test 2: only scale (1.2x1.2)")
    cmd2 = [
        sys.executable, "image_transform.py", test_image,
        "--scale", "1.2", "1.2",
        "--output", "test_scale_only.tif"
    ]
    subprocess.run(cmd2)
    print()
    
    # test 3: only translation
    print("test 3: only translation (dx=50, dy=-30)")
    cmd3 = [
        sys.executable, "image_transform.py", test_image,
        "--dx", "50", "--dy", "-30",
        "--output", "test_translation_only.tif"
    ]
    subprocess.run(cmd3)
    print()
    
    # test 4: only rotation
    print("test 4: only rotation (15 degrees)")
    cmd4 = [
        sys.executable, "image_transform.py", test_image,
        "--rotation", "15",
        "--output", "test_rotation_only.tif"
    ]
    subprocess.run(cmd4)
    print()
    
    # test 5: combined transformation
    print("test 5: combined transformation (resize + scale + translation + rotation)")
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
    
    print("=== test completed ===")
    print("generated files:")
    print("  - test_resize_only.tif")
    print("  - test_scale_only.tif")
    print("  - test_translation_only.tif")
    print("  - test_rotation_only.tif")
    print("  - test_combined.tif")

def show_usage_examples():
    """show usage examples"""
    print("=== usage examples ===\n")
    
    examples = [
        {
            "description": "only resize to 0.5x",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.5 0.5"
        },
        {
            "description": "only scale to 1.2x",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.2 1.2"
        },
        {
            "description": "only translation (dx=50, dy=-30)",
            "command": "python image_transform.py section_30_r01_c01.tif --dx 50 --dy -30"
        },
        {
            "description": "only rotation (15 degrees)",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 15"
        },
        {
            "description": "combined transformation",
            "command": "python image_transform.py section_30_r01_c01.tif --resize 0.8 0.8 --scale 1.2 --dx 50 --dy -30 --rotation 15"
        },
        {
            "description": "specify rotation center point",
            "command": "python image_transform.py section_30_r01_c01.tif --rotation 45 --center 512 512"
        },
        {
            "description": "specify output file name",
            "command": "python image_transform.py section_30_r01_c01.tif --scale 1.5 --output my_transformed_image.tif"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   {example['command']}")
        print()

if __name__ == "__main__":
    print("image transform tool demo\n")
    
    # show usage examples
    show_usage_examples()
    
    # ask if run test
    response = input("run test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_transform_test()
    else:
        print("skip test.") 
 
 
 
 
 
 
