#!/usr/bin/env python3
"""
stack_images_1k.py - Stack images to 1Kx1K resolution TIFF
========================================================
Input
-----
1. linear_order.txt   (section order from linear_order_builder.py)
2. image_dir/         (directory with section images)
3. output_file        (output stacked volume file)

Output
------
- 3D volume as .tif file with 1Kx1K resolution

Run
---
python stack_images_1k.py linear_order_ssim.txt w7_png_4k/ --output stacked_volume_1k.tif
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
    print("Error: tifffile is required for this script")
    exit(1)

def load_order_file(order_file: Path) -> List[str]:
    """Load section order from file."""
    with open(order_file, 'r') as f:
        content = f.read().strip()
    return content.split()

def load_and_resize_image(image_path: Path, target_size: int = 1024) -> Optional[np.ndarray]:
    """Load image, resize to target size, and convert to grayscale if needed."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None
        
        # Resize to target size
        if img.shape != (target_size, target_size):
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def stack_images_1k(order: List[str], image_dir: Path, output_file: Path, 
                   target_size: int = 1024, show_progress: bool = True) -> bool:
    """Stack images according to the order and resize to 1Kx1K."""
    
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
        if order_section in available_images:
            section_mapping[order_section] = order_section
        else:
            print(f"Warning: No image found for {order_section}")
    
    if not section_mapping:
        print("Error: No matching images found!")
        return False
    
    print(f"Found {len(section_mapping)} matching images")
    print(f"Target resolution: {target_size}x{target_size}")
    
    # Load first image to get dimensions
    first_section = order[0]
    if first_section not in section_mapping:
        print(f"Error: First section {first_section} not found in image directory")
        return False
    
    first_img = load_and_resize_image(available_images[section_mapping[first_section]], target_size)
    if first_img is None:
        return False
    
    height, width = first_img.shape
    print(f"Resized image dimensions: {width}x{height}")
    
    # Initialize 3D volume
    volume = np.zeros((len(order), height, width), dtype=np.uint8)
    
    # Load images in order
    loaded_count = 0
    missing_count = 0
    
    for i, section_name in enumerate(order):
        if show_progress and i % 10 == 0:
            print(f"Loading and resizing image {i+1}/{len(order)}: {section_name}")
        
        if section_name in section_mapping:
            img = load_and_resize_image(available_images[section_mapping[section_name]], target_size)
            if img is not None:
                volume[i] = img
                loaded_count += 1
            else:
                missing_count += 1
                print(f"Warning: Failed to load {section_name}")
        else:
            missing_count += 1
            print(f"Warning: Image for {section_name} not found")
    
    print(f"Successfully loaded and resized {loaded_count}/{len(order)} images")
    if missing_count > 0:
        print(f"Missing {missing_count} images")
    
    # Save as TIFF
    print(f"Saving volume as TIFF: {volume.shape}, {volume.dtype}")
    tifffile.imwrite(output_file, volume, compression='lzw')
    print(f"Saved volume to {output_file}")
    
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
    ap = argparse.ArgumentParser(description='Stack images to 1Kx1K resolution TIFF')
    ap.add_argument('order_file', help='File with section order (from linear_order_builder.py)')
    ap.add_argument('image_dir', help='Directory containing section images')
    ap.add_argument('--output', default='stacked_volume_1k.tif', help='Output volume file')
    ap.add_argument('--size', type=int, default=1024, help='Target resolution (default: 1024)')
    ap.add_argument('--viz', help='Output visualization file')
    ap.add_argument('--no-progress', action='store_true', help='Hide progress messages')
    args = ap.parse_args()
    
    # Load order
    order = load_order_file(Path(args.order_file))
    print(f"Loaded order with {len(order)} sections")
    
    # Stack images
    success = stack_images_1k(
        order, 
        Path(args.image_dir), 
        Path(args.output), 
        args.size,
        not args.no_progress
    )
    
    if success and args.viz:
        # Load the volume for visualization
        volume = tifffile.imread(args.output)
        create_visualization(volume, Path(args.viz))
    
    if success:
        print("Image stacking to 1Kx1K completed successfully!")
    else:
        print("Image stacking failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 
 
 
 
 
 
 
 
 
 
 
 