#!/usr/bin/env python
"""Build a multi-page TIFF stack from PNG sections in a specified order.

Usage:
    python build_stack.py output.tif
Requires Pillow.
"""
import sys, os
from PIL import Image
from feabas import storage

ORDER = [
"w06_dSection_002_r01_c01",
"w06_dSection_012_r01_c01",
"w06_dSection_018_r01_c01",
"w06_dSection_015_r01_c01",
"w06_dSection_003_r01_c01",
"w06_dSection_013_r01_c01",
"w06_dSection_042_r01_c01",
"w06_dSection_029_r01_c01",
"w06_dSection_006_r01_c01",
"w06_dSection_046_r01_c01",
"w06_dSection_033_r01_c01",
"w06_dSection_045_r01_c01",
"w06_dSection_024_r01_c01",
"w06_dSection_019_r01_c01",
"w06_dSection_053_r01_c01",
"w06_dSection_020_r01_c01",
"w06_dSection_036_r01_c01",
"w06_dSection_039_r01_c01",
"w06_dSection_043_r01_c01",
"w06_dSection_030_r01_c01",
"w06_dSection_060_r01_c01",
"w06_dSection_007_r01_c01",
"w06_dSection_034_r01_c01",
"w06_dSection_008_r01_c01",
"w06_dSection_035_r01_c01",
"w06_dSection_031_r01_c01",
"w06_dSection_025_r01_c01",
"w06_dSection_009_r01_c01",
"w06_dSection_041_r01_c01",
"w06_dSection_017_r01_c01",
"w06_dSection_055_r01_c01",
"w06_dSection_058_r01_c01",
"w06_dSection_040_r01_c01",
"w06_dSection_010_r01_c01",
"w06_dSection_048_r01_c01",
"w06_dSection_051_r01_c01",
"w06_dSection_032_r01_c01",
"w06_dSection_028_r01_c01",
"w06_dSection_047_r01_c01",
"w06_dSection_014_r01_c01",
"w06_dSection_052_r01_c01",
"w06_dSection_011_r01_c01",
"w06_dSection_005_r01_c01",
"w06_dSection_044_r01_c01",
"w06_dSection_037_r01_c01",
"w06_dSection_022_r01_c01",
"w06_dSection_021_r01_c01",
"w06_dSection_004_r01_c01",
"w06_dSection_049_r01_c01",
"w06_dSection_026_r01_c01",
"w06_dSection_050_r01_c01",
"w06_dSection_056_r01_c01",
"w06_dSection_038_r01_c01",
"w06_dSection_057_r01_c01",
"w06_dSection_016_r01_c01",
"w06_dSection_023_r01_c01",
"w06_dSection_054_r01_c01",
"w06_dSection_027_r01_c01",
"w06_dSection_059_r01_c01",
]

ROOT_DIR = os.path.join("stitched_sections", "mip0")


def main(outname: str):
    imgs = []
    for sec in ORDER:
        img_path = os.path.join(ROOT_DIR, sec, f"{sec}.png")
        if not storage.file_exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("L")
        imgs.append(img)
    base = imgs[0]
    base.save(outname, save_all=True, append_images=imgs[1:])
    print(f"Saved stack to {outname} with {len(imgs)} slices.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_stack.py output.tif")
        sys.exit(1)
    main(sys.argv[1]) 
"""Build a multi-page TIFF stack from PNG sections in a specified order.

Usage:
    python build_stack.py output.tif
Requires Pillow.
"""
import sys, os
from PIL import Image
from feabas import storage

ORDER = [
"w06_dSection_002_r01_c01",
"w06_dSection_012_r01_c01",
"w06_dSection_018_r01_c01",
"w06_dSection_015_r01_c01",
"w06_dSection_003_r01_c01",
"w06_dSection_013_r01_c01",
"w06_dSection_042_r01_c01",
"w06_dSection_029_r01_c01",
"w06_dSection_006_r01_c01",
"w06_dSection_046_r01_c01",
"w06_dSection_033_r01_c01",
"w06_dSection_045_r01_c01",
"w06_dSection_024_r01_c01",
"w06_dSection_019_r01_c01",
"w06_dSection_053_r01_c01",
"w06_dSection_020_r01_c01",
"w06_dSection_036_r01_c01",
"w06_dSection_039_r01_c01",
"w06_dSection_043_r01_c01",
"w06_dSection_030_r01_c01",
"w06_dSection_060_r01_c01",
"w06_dSection_007_r01_c01",
"w06_dSection_034_r01_c01",
"w06_dSection_008_r01_c01",
"w06_dSection_035_r01_c01",
"w06_dSection_031_r01_c01",
"w06_dSection_025_r01_c01",
"w06_dSection_009_r01_c01",
"w06_dSection_041_r01_c01",
"w06_dSection_017_r01_c01",
"w06_dSection_055_r01_c01",
"w06_dSection_058_r01_c01",
"w06_dSection_040_r01_c01",
"w06_dSection_010_r01_c01",
"w06_dSection_048_r01_c01",
"w06_dSection_051_r01_c01",
"w06_dSection_032_r01_c01",
"w06_dSection_028_r01_c01",
"w06_dSection_047_r01_c01",
"w06_dSection_014_r01_c01",
"w06_dSection_052_r01_c01",
"w06_dSection_011_r01_c01",
"w06_dSection_005_r01_c01",
"w06_dSection_044_r01_c01",
"w06_dSection_037_r01_c01",
"w06_dSection_022_r01_c01",
"w06_dSection_021_r01_c01",
"w06_dSection_004_r01_c01",
"w06_dSection_049_r01_c01",
"w06_dSection_026_r01_c01",
"w06_dSection_050_r01_c01",
"w06_dSection_056_r01_c01",
"w06_dSection_038_r01_c01",
"w06_dSection_057_r01_c01",
"w06_dSection_016_r01_c01",
"w06_dSection_023_r01_c01",
"w06_dSection_054_r01_c01",
"w06_dSection_027_r01_c01",
"w06_dSection_059_r01_c01",
]

ROOT_DIR = os.path.join("stitched_sections", "mip0")


def main(outname: str):
    imgs = []
    for sec in ORDER:
        img_path = os.path.join(ROOT_DIR, sec, f"{sec}.png")
        if not storage.file_exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("L")
        imgs.append(img)
    base = imgs[0]
    base.save(outname, save_all=True, append_images=imgs[1:])
    print(f"Saved stack to {outname} with {len(imgs)} slices.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_stack.py output.tif")
        sys.exit(1)
    main(sys.argv[1]) 
 
"""Build a multi-page TIFF stack from PNG sections in a specified order.

Usage:
    python build_stack.py output.tif
Requires Pillow.
"""
import sys, os
from PIL import Image
from feabas import storage

ORDER = [
"w06_dSection_002_r01_c01",
"w06_dSection_012_r01_c01",
"w06_dSection_018_r01_c01",
"w06_dSection_015_r01_c01",
"w06_dSection_003_r01_c01",
"w06_dSection_013_r01_c01",
"w06_dSection_042_r01_c01",
"w06_dSection_029_r01_c01",
"w06_dSection_006_r01_c01",
"w06_dSection_046_r01_c01",
"w06_dSection_033_r01_c01",
"w06_dSection_045_r01_c01",
"w06_dSection_024_r01_c01",
"w06_dSection_019_r01_c01",
"w06_dSection_053_r01_c01",
"w06_dSection_020_r01_c01",
"w06_dSection_036_r01_c01",
"w06_dSection_039_r01_c01",
"w06_dSection_043_r01_c01",
"w06_dSection_030_r01_c01",
"w06_dSection_060_r01_c01",
"w06_dSection_007_r01_c01",
"w06_dSection_034_r01_c01",
"w06_dSection_008_r01_c01",
"w06_dSection_035_r01_c01",
"w06_dSection_031_r01_c01",
"w06_dSection_025_r01_c01",
"w06_dSection_009_r01_c01",
"w06_dSection_041_r01_c01",
"w06_dSection_017_r01_c01",
"w06_dSection_055_r01_c01",
"w06_dSection_058_r01_c01",
"w06_dSection_040_r01_c01",
"w06_dSection_010_r01_c01",
"w06_dSection_048_r01_c01",
"w06_dSection_051_r01_c01",
"w06_dSection_032_r01_c01",
"w06_dSection_028_r01_c01",
"w06_dSection_047_r01_c01",
"w06_dSection_014_r01_c01",
"w06_dSection_052_r01_c01",
"w06_dSection_011_r01_c01",
"w06_dSection_005_r01_c01",
"w06_dSection_044_r01_c01",
"w06_dSection_037_r01_c01",
"w06_dSection_022_r01_c01",
"w06_dSection_021_r01_c01",
"w06_dSection_004_r01_c01",
"w06_dSection_049_r01_c01",
"w06_dSection_026_r01_c01",
"w06_dSection_050_r01_c01",
"w06_dSection_056_r01_c01",
"w06_dSection_038_r01_c01",
"w06_dSection_057_r01_c01",
"w06_dSection_016_r01_c01",
"w06_dSection_023_r01_c01",
"w06_dSection_054_r01_c01",
"w06_dSection_027_r01_c01",
"w06_dSection_059_r01_c01",
]

ROOT_DIR = os.path.join("stitched_sections", "mip0")


def main(outname: str):
    imgs = []
    for sec in ORDER:
        img_path = os.path.join(ROOT_DIR, sec, f"{sec}.png")
        if not storage.file_exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("L")
        imgs.append(img)
    base = imgs[0]
    base.save(outname, save_all=True, append_images=imgs[1:])
    print(f"Saved stack to {outname} with {len(imgs)} slices.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_stack.py output.tif")
        sys.exit(1)
    main(sys.argv[1]) 
"""Build a multi-page TIFF stack from PNG sections in a specified order.

Usage:
    python build_stack.py output.tif
Requires Pillow.
"""
import sys, os
from PIL import Image
from feabas import storage

ORDER = [
"w06_dSection_002_r01_c01",
"w06_dSection_012_r01_c01",
"w06_dSection_018_r01_c01",
"w06_dSection_015_r01_c01",
"w06_dSection_003_r01_c01",
"w06_dSection_013_r01_c01",
"w06_dSection_042_r01_c01",
"w06_dSection_029_r01_c01",
"w06_dSection_006_r01_c01",
"w06_dSection_046_r01_c01",
"w06_dSection_033_r01_c01",
"w06_dSection_045_r01_c01",
"w06_dSection_024_r01_c01",
"w06_dSection_019_r01_c01",
"w06_dSection_053_r01_c01",
"w06_dSection_020_r01_c01",
"w06_dSection_036_r01_c01",
"w06_dSection_039_r01_c01",
"w06_dSection_043_r01_c01",
"w06_dSection_030_r01_c01",
"w06_dSection_060_r01_c01",
"w06_dSection_007_r01_c01",
"w06_dSection_034_r01_c01",
"w06_dSection_008_r01_c01",
"w06_dSection_035_r01_c01",
"w06_dSection_031_r01_c01",
"w06_dSection_025_r01_c01",
"w06_dSection_009_r01_c01",
"w06_dSection_041_r01_c01",
"w06_dSection_017_r01_c01",
"w06_dSection_055_r01_c01",
"w06_dSection_058_r01_c01",
"w06_dSection_040_r01_c01",
"w06_dSection_010_r01_c01",
"w06_dSection_048_r01_c01",
"w06_dSection_051_r01_c01",
"w06_dSection_032_r01_c01",
"w06_dSection_028_r01_c01",
"w06_dSection_047_r01_c01",
"w06_dSection_014_r01_c01",
"w06_dSection_052_r01_c01",
"w06_dSection_011_r01_c01",
"w06_dSection_005_r01_c01",
"w06_dSection_044_r01_c01",
"w06_dSection_037_r01_c01",
"w06_dSection_022_r01_c01",
"w06_dSection_021_r01_c01",
"w06_dSection_004_r01_c01",
"w06_dSection_049_r01_c01",
"w06_dSection_026_r01_c01",
"w06_dSection_050_r01_c01",
"w06_dSection_056_r01_c01",
"w06_dSection_038_r01_c01",
"w06_dSection_057_r01_c01",
"w06_dSection_016_r01_c01",
"w06_dSection_023_r01_c01",
"w06_dSection_054_r01_c01",
"w06_dSection_027_r01_c01",
"w06_dSection_059_r01_c01",
]

ROOT_DIR = os.path.join("stitched_sections", "mip0")


def main(outname: str):
    imgs = []
    for sec in ORDER:
        img_path = os.path.join(ROOT_DIR, sec, f"{sec}.png")
        if not storage.file_exists(img_path):
            raise FileNotFoundError(img_path)
        img = Image.open(img_path).convert("L")
        imgs.append(img)
    base = imgs[0]
    base.save(outname, save_all=True, append_images=imgs[1:])
    print(f"Saved stack to {outname} with {len(imgs)} slices.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_stack.py output.tif")
        sys.exit(1)
    main(sys.argv[1]) 
 
 
 
 
 
 
 
 
 
 
 
 
 