#!/usr/bin/env python3
"""
test auto mask generator
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from section_counter import process_image_with_auto_generator, process_image_with_sam

def test_auto_generator():
    """test auto mask generator"""
    
    # check if there are test images
    test_images = [
        "1750864145001.jpg",  # images in your directory
        "w06_250nm_1.0mms_60sections_RGB.png_files.png"  # images in the current project
    ]
    
    for img_name in test_images:
        if os.path.exists(img_name):
            print(f"🔍 test image: {img_name}")
            
            # test auto mask generator
            print("📊 test auto mask generator mode...")
            try:
                result_auto = process_image_with_auto_generator(img_name, 'vit_l')
                if result_auto:
                    print("✅ auto mask generator test passed")
                else:
                    print("❌ auto mask generator test failed")
            except Exception as e:
                print(f"❌ auto mask generator error: {e}")
            
            # test original grid point mode
            print("📊 test improved grid point mode...")
            try:
                result_grid = process_image_with_sam(img_name, 'vit_l', use_auto_generator=False)
                if result_grid:
                    print("✅ improved grid point mode test passed")
                else:
                    print("❌ improved grid point mode test failed")
            except Exception as e:
                print(f"❌ improved grid point mode error: {e}")
            
            break
    else:
        print("❌ test image not found")
        print("please ensure there is one of the following images:")
        for img in test_images:
            print(f"  - {img}")

if __name__ == "__main__":
    test_auto_generator() 
