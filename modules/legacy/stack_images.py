#!/usr/bin/env python3
"""
stack_images.py - Stack images according to linear order
======================================================
Input
-----
1. linear_order.txt   (section order from linear_order_builder.py)
2. image_dir/         (directory with section images)
3. output_file        (output stacked volume file)

Output
------
- 3D volume as .npy or .tif file
- Optional visualization of the stack

Run
---
python stack_images.py linear_order_ssim.txt rotated_heatmaps_gray_patches_gray/ --output stacked_volume.npy
"""

import argparse
import numpy as np
from pathlib import Path
import cv2
from typing import List, Optional
import matplotlib.pyplot as plt

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False
    print("Warning: tifffile not available, .tif output disabled")

def load_order_file(order_file: Path) -> List[str]:
    """Load section order from file."""
    with open(order_file, 'r') as f:
        content = f.read().strip()
    return content.split()

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load image and convert to grayscale if needed."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def extract_section_number(section_name: str) -> str:
    """Extract section number from section name."""
    # For w7_png_4k directory, the filenames match exactly
    # like "section_44_r01_c01.png" matches "section_44_r01_c01"
    return section_name

def stack_images(order: List[str], image_dir: Path, output_file: Path, 
                format: str = 'npy', show_progress: bool = True) -> bool:
    """Stack images according to the order."""
    
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist")
        return False
    
    # Find all available images and create mapping
    available_images = {}
    section_mapping = {}  # Map from order names to actual filenames
    
    for img_file in image_dir.glob("*.png"):
        section_name = img_file.stem
        available_images[section_name] = img_file
    
    print(f"Found {len(available_images)} images in {image_dir}")
    print(f"Order contains {len(order)} sections")
    
    # Create mapping from order names to available image names
    for order_section in order:
        section_number = extract_section_number(order_section)
        if section_number in available_images:
            section_mapping[order_section] = section_number
        else:
            print(f"Warning: No image found for {order_section} (extracted: {section_number})")
    
    if not section_mapping:
        print("Error: No matching images found!")
        return False
    
    print(f"Found {len(section_mapping)} matching images")
    
    # Load first image to get dimensions
    first_section = order[0]
    if first_section not in section_mapping:
        print(f"Error: First section {first_section} not found in image directory")
        return False
    
    first_img = load_image(available_images[section_mapping[first_section]])
    if first_img is None:
        return False
    
    height, width = first_img.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Initialize 3D volume
    volume = np.zeros((len(order), height, width), dtype=np.uint8)
    
    # Load images in order
    loaded_count = 0
    missing_count = 0
    
    for i, section_name in enumerate(order):
        if show_progress and i % 10 == 0:
            print(f"Loading image {i+1}/{len(order)}: {section_name}")
        
        if section_name in section_mapping:
            img = load_image(available_images[section_mapping[section_name]])
            if img is not None:
                volume[i] = img
                loaded_count += 1
            else:
                missing_count += 1
                print(f"Warning: Failed to load {section_name}")
        else:
            missing_count += 1
            print(f"Warning: Image for {section_name} not found")
    
    print(f"Successfully loaded {loaded_count}/{len(order)} images")
    if missing_count > 0:
        print(f"Missing {missing_count} images")
    
    # Save volume
    if format.lower() == 'npy':
        np.save(output_file, volume)
        print(f"Saved volume to {output_file}")
    elif format.lower() == 'tif' and HAS_TIFF:
        tifffile.imwrite(output_file, volume)
        print(f"Saved volume to {output_file}")
    else:
        print(f"Unsupported format: {format}")
        return False
    
    return True

def create_visualization(volume: np.ndarray, output_file: Path):
    """Create visualization of the stacked volume."""
    try:
        # Create a montage view
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY slice (middle)
        mid_z = volume.shape[0] // 2
        axes[0, 0].imshow(volume[mid_z], cmap='gray')
        axes[0, 0].set_title(f'XY Slice (z={mid_z})')
        axes[0, 0].axis('off')
        
        # XZ slice (middle)
        mid_y = volume.shape[1] // 2
        axes[0, 1].imshow(volume[:, mid_y, :], cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'XZ Slice (y={mid_y})')
        axes[0, 1].axis('off')
        
        # YZ slice (middle)
        mid_x = volume.shape[2] // 2
        axes[1, 0].imshow(volume[:, :, mid_x], cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'YZ Slice (x={mid_x})')
        axes[1, 0].axis('off')
        
        # Volume statistics
        axes[1, 1].text(0.1, 0.8, f'Volume shape: {volume.shape}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Data type: {volume.dtype}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Value range: [{volume.min()}, {volume.max()}]', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Mean value: {volume.mean():.1f}', fontsize=12)
        axes[1, 1].set_title('Volume Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_file}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    ap = argparse.ArgumentParser(description='Stack images according to linear order')
    ap.add_argument('order_file', help='File with section order (from linear_order_builder.py)')
    ap.add_argument('image_dir', help='Directory containing section images')
    ap.add_argument('--output', default='stacked_volume.npy', help='Output volume file')
    ap.add_argument('--format', choices=['npy', 'tif'], default='npy', help='Output format')
    ap.add_argument('--viz', help='Output visualization file')
    ap.add_argument('--no-progress', action='store_true', help='Hide progress messages')
    args = ap.parse_args()
    
    # Load order
    order = load_order_file(Path(args.order_file))
    print(f"Loaded order with {len(order)} sections")
    
    # Stack images
    success = stack_images(
        order, 
        Path(args.image_dir), 
        Path(args.output), 
        args.format, 
        not args.no_progress
    )
    
    if success and args.viz:
        # Load the volume for visualization
        if args.format == 'npy':
            volume = np.load(args.output)
        else:
            volume = tifffile.imread(args.output)
        
        create_visualization(volume, Path(args.viz))
    
    if success:
        print("Image stacking completed successfully!")
    else:
        print("Image stacking failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 
"""
stack_images.py - Stack images according to linear order
======================================================
Input
-----
1. linear_order.txt   (section order from linear_order_builder.py)
2. image_dir/         (directory with section images)
3. output_file        (output stacked volume file)

Output
------
- 3D volume as .npy or .tif file
- Optional visualization of the stack

Run
---
python stack_images.py linear_order_ssim.txt rotated_heatmaps_gray_patches_gray/ --output stacked_volume.npy
"""

import argparse
import numpy as np
from pathlib import Path
import cv2
from typing import List, Optional
import matplotlib.pyplot as plt

try:
    import tifffile
    HAS_TIFF = True
except ImportError:
    HAS_TIFF = False
    print("Warning: tifffile not available, .tif output disabled")

def load_order_file(order_file: Path) -> List[str]:
    """Load section order from file."""
    with open(order_file, 'r') as f:
        content = f.read().strip()
    return content.split()

def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load image and convert to grayscale if needed."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def extract_section_number(section_name: str) -> str:
    """Extract section number from section name."""
    # For w7_png_4k directory, the filenames match exactly
    # like "section_44_r01_c01.png" matches "section_44_r01_c01"
    return section_name

def stack_images(order: List[str], image_dir: Path, output_file: Path, 
                format: str = 'npy', show_progress: bool = True) -> bool:
    """Stack images according to the order."""
    
    if not image_dir.exists():
        print(f"Error: Image directory {image_dir} does not exist")
        return False
    
    # Find all available images and create mapping
    available_images = {}
    section_mapping = {}  # Map from order names to actual filenames
    
    for img_file in image_dir.glob("*.png"):
        section_name = img_file.stem
        available_images[section_name] = img_file
    
    print(f"Found {len(available_images)} images in {image_dir}")
    print(f"Order contains {len(order)} sections")
    
    # Create mapping from order names to available image names
    for order_section in order:
        section_number = extract_section_number(order_section)
        if section_number in available_images:
            section_mapping[order_section] = section_number
        else:
            print(f"Warning: No image found for {order_section} (extracted: {section_number})")
    
    if not section_mapping:
        print("Error: No matching images found!")
        return False
    
    print(f"Found {len(section_mapping)} matching images")
    
    # Load first image to get dimensions
    first_section = order[0]
    if first_section not in section_mapping:
        print(f"Error: First section {first_section} not found in image directory")
        return False
    
    first_img = load_image(available_images[section_mapping[first_section]])
    if first_img is None:
        return False
    
    height, width = first_img.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Initialize 3D volume
    volume = np.zeros((len(order), height, width), dtype=np.uint8)
    
    # Load images in order
    loaded_count = 0
    missing_count = 0
    
    for i, section_name in enumerate(order):
        if show_progress and i % 10 == 0:
            print(f"Loading image {i+1}/{len(order)}: {section_name}")
        
        if section_name in section_mapping:
            img = load_image(available_images[section_mapping[section_name]])
            if img is not None:
                volume[i] = img
                loaded_count += 1
            else:
                missing_count += 1
                print(f"Warning: Failed to load {section_name}")
        else:
            missing_count += 1
            print(f"Warning: Image for {section_name} not found")
    
    print(f"Successfully loaded {loaded_count}/{len(order)} images")
    if missing_count > 0:
        print(f"Missing {missing_count} images")
    
    # Save volume
    if format.lower() == 'npy':
        np.save(output_file, volume)
        print(f"Saved volume to {output_file}")
    elif format.lower() == 'tif' and HAS_TIFF:
        tifffile.imwrite(output_file, volume)
        print(f"Saved volume to {output_file}")
    else:
        print(f"Unsupported format: {format}")
        return False
    
    return True

def create_visualization(volume: np.ndarray, output_file: Path):
    """Create visualization of the stacked volume."""
    try:
        # Create a montage view
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY slice (middle)
        mid_z = volume.shape[0] // 2
        axes[0, 0].imshow(volume[mid_z], cmap='gray')
        axes[0, 0].set_title(f'XY Slice (z={mid_z})')
        axes[0, 0].axis('off')
        
        # XZ slice (middle)
        mid_y = volume.shape[1] // 2
        axes[0, 1].imshow(volume[:, mid_y, :], cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'XZ Slice (y={mid_y})')
        axes[0, 1].axis('off')
        
        # YZ slice (middle)
        mid_x = volume.shape[2] // 2
        axes[1, 0].imshow(volume[:, :, mid_x], cmap='gray', aspect='auto')
        axes[1, 0].set_title(f'YZ Slice (x={mid_x})')
        axes[1, 0].axis('off')
        
        # Volume statistics
        axes[1, 1].text(0.1, 0.8, f'Volume shape: {volume.shape}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Data type: {volume.dtype}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Value range: [{volume.min()}, {volume.max()}]', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Mean value: {volume.mean():.1f}', fontsize=12)
        axes[1, 1].set_title('Volume Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_file}")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    ap = argparse.ArgumentParser(description='Stack images according to linear order')
    ap.add_argument('order_file', help='File with section order (from linear_order_builder.py)')
    ap.add_argument('image_dir', help='Directory containing section images')
    ap.add_argument('--output', default='stacked_volume.npy', help='Output volume file')
    ap.add_argument('--format', choices=['npy', 'tif'], default='npy', help='Output format')
    ap.add_argument('--viz', help='Output visualization file')
    ap.add_argument('--no-progress', action='store_true', help='Hide progress messages')
    args = ap.parse_args()
    
    # Load order
    order = load_order_file(Path(args.order_file))
    print(f"Loaded order with {len(order)} sections")
    
    # Stack images
    success = stack_images(
        order, 
        Path(args.image_dir), 
        Path(args.output), 
        args.format, 
        not args.no_progress
    )
    
    if success and args.viz:
        # Load the volume for visualization
        if args.format == 'npy':
            volume = np.load(args.output)
        else:
            volume = tifffile.imread(args.output)
        
        create_visualization(volume, Path(args.viz))
    
    if success:
        print("Image stacking completed successfully!")
    else:
        print("Image stacking failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 
 
 
 
 
 
 
 
 
 
 
 
 