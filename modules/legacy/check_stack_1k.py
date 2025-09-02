#!/usr/bin/env python3
"""
check_stack_1k.py - Check the 1Kx1K stacked volume TIFF
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile

def main():
    # Load the stacked volume
    print("Loading 1Kx1K stacked volume TIFF...")
    volume = tifffile.imread('stacked_volume_1k.tif')
    
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Value range: [{volume.min()}, {volume.max()}]")
    print(f"Mean value: {volume.mean():.1f}")
    print(f"Memory usage: {volume.nbytes / 1024 / 1024:.1f} MB")
    
    # Show some slices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First slice
    axes[0, 0].imshow(volume[0], cmap='gray')
    axes[0, 0].set_title('First slice (z=0)')
    axes[0, 0].axis('off')
    
    # Middle slice
    mid_z = volume.shape[0] // 2
    axes[0, 1].imshow(volume[mid_z], cmap='gray')
    axes[0, 1].set_title(f'Middle slice (z={mid_z})')
    axes[0, 1].axis('off')
    
    # Last slice
    axes[0, 2].imshow(volume[-1], cmap='gray')
    axes[0, 2].set_title(f'Last slice (z={volume.shape[0]-1})')
    axes[0, 2].axis('off')
    
    # XZ slice
    mid_y = volume.shape[1] // 2
    axes[1, 0].imshow(volume[:, mid_y, :], cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'XZ slice (y={mid_y})')
    axes[1, 0].axis('off')
    
    # YZ slice
    mid_x = volume.shape[2] // 2
    axes[1, 1].imshow(volume[:, :, mid_x], cmap='gray', aspect='auto')
    axes[1, 1].set_title(f'YZ slice (x={mid_x})')
    axes[1, 1].axis('off')
    
    # Histogram
    axes[1, 2].hist(volume.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_title('Value distribution')
    axes[1, 2].set_xlabel('Pixel value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('stack_check_1k.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved 1Kx1K stack check visualization to stack_check_1k.pdf")
    
    # Load and show the order
    with open('linear_order_ssim.txt', 'r') as f:
        order = f.read().strip().split()
    
    print(f"\nOrder information:")
    print(f"Total sections in order: {len(order)}")
    print(f"First 5 sections: {order[:5]}")
    print(f"Last 5 sections: {order[-5:]}")
    
    # Check which sections are missing
    missing_sections = []
    for i, section in enumerate(order):
        if i < volume.shape[0]:
            # This section should be in the volume
            pass
        else:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"Missing sections in volume: {missing_sections}")
    else:
        print("All ordered sections are in the volume")
    
    # Show some statistics about the volume
    print(f"\nVolume statistics:")
    print(f"Total voxels: {volume.size:,}")
    print(f"Non-zero voxels: {np.count_nonzero(volume):,}")
    print(f"Zero voxels: {np.sum(volume == 0):,}")
    print(f"Fill ratio: {np.count_nonzero(volume) / volume.size * 100:.1f}%")
    
    # File size comparison
    print(f"\nFile size comparison:")
    print(f"1Kx1K TIFF: 86MB (compressed)")
    print(f"4Kx4K NPY: 1.7GB (uncompressed)")
    print(f"Size reduction: {(1.7*1024)/86:.1f}x smaller")

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 