#!/usr/bin/env python3
"""
check_stack_4k.py - Check the 4k stacked volume data
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the stacked volume
    print("Loading 4k stacked volume...")
    volume = np.load('stacked_volume_4k.npy')
    
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Value range: [{volume.min()}, {volume.max()}]")
    print(f"Mean value: {volume.mean():.1f}")
    print(f"Memory usage: {volume.nbytes / 1024 / 1024 / 1024:.1f} GB")
    
    # Show some slices (downsampled for visualization)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First slice (downsampled)
    first_slice = volume[0][::8, ::8]  # Downsample by 8x
    axes[0, 0].imshow(first_slice, cmap='gray')
    axes[0, 0].set_title('First slice (z=0, downsampled 8x)')
    axes[0, 0].axis('off')
    
    # Middle slice (downsampled)
    mid_z = volume.shape[0] // 2
    mid_slice = volume[mid_z][::8, ::8]  # Downsample by 8x
    axes[0, 1].imshow(mid_slice, cmap='gray')
    axes[0, 1].set_title(f'Middle slice (z={mid_z}, downsampled 8x)')
    axes[0, 1].axis('off')
    
    # Last slice (downsampled)
    last_slice = volume[-1][::8, ::8]  # Downsample by 8x
    axes[0, 2].imshow(last_slice, cmap='gray')
    axes[0, 2].set_title(f'Last slice (z={volume.shape[0]-1}, downsampled 8x)')
    axes[0, 2].axis('off')
    
    # XZ slice (downsampled)
    mid_y = volume.shape[1] // 2
    xz_slice = volume[:, mid_y, ::8]  # Downsample X by 8x
    axes[1, 0].imshow(xz_slice, cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'XZ slice (y={mid_y}, X downsampled 8x)')
    axes[1, 0].axis('off')
    
    # YZ slice (downsampled)
    mid_x = volume.shape[2] // 2
    yz_slice = volume[:, ::8, mid_x]  # Downsample Y by 8x
    axes[1, 1].imshow(yz_slice, cmap='gray', aspect='auto')
    axes[1, 1].set_title(f'YZ slice (x={mid_x}, Y downsampled 8x)')
    axes[1, 1].axis('off')
    
    # Histogram
    # Sample a subset for histogram to avoid memory issues
    sample_data = volume[::4, ::16, ::16].flatten()  # Sample every 4th slice, 16x downsampled
    axes[1, 2].hist(sample_data, bins=50, alpha=0.7)
    axes[1, 2].set_title('Value distribution (sampled)')
    axes[1, 2].set_xlabel('Pixel value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('stack_check_4k.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved 4k stack check visualization to stack_check_4k.pdf")
    
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

if __name__ == '__main__':
    main() 
"""
check_stack_4k.py - Check the 4k stacked volume data
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the stacked volume
    print("Loading 4k stacked volume...")
    volume = np.load('stacked_volume_4k.npy')
    
    print(f"Volume shape: {volume.shape}")
    print(f"Data type: {volume.dtype}")
    print(f"Value range: [{volume.min()}, {volume.max()}]")
    print(f"Mean value: {volume.mean():.1f}")
    print(f"Memory usage: {volume.nbytes / 1024 / 1024 / 1024:.1f} GB")
    
    # Show some slices (downsampled for visualization)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # First slice (downsampled)
    first_slice = volume[0][::8, ::8]  # Downsample by 8x
    axes[0, 0].imshow(first_slice, cmap='gray')
    axes[0, 0].set_title('First slice (z=0, downsampled 8x)')
    axes[0, 0].axis('off')
    
    # Middle slice (downsampled)
    mid_z = volume.shape[0] // 2
    mid_slice = volume[mid_z][::8, ::8]  # Downsample by 8x
    axes[0, 1].imshow(mid_slice, cmap='gray')
    axes[0, 1].set_title(f'Middle slice (z={mid_z}, downsampled 8x)')
    axes[0, 1].axis('off')
    
    # Last slice (downsampled)
    last_slice = volume[-1][::8, ::8]  # Downsample by 8x
    axes[0, 2].imshow(last_slice, cmap='gray')
    axes[0, 2].set_title(f'Last slice (z={volume.shape[0]-1}, downsampled 8x)')
    axes[0, 2].axis('off')
    
    # XZ slice (downsampled)
    mid_y = volume.shape[1] // 2
    xz_slice = volume[:, mid_y, ::8]  # Downsample X by 8x
    axes[1, 0].imshow(xz_slice, cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'XZ slice (y={mid_y}, X downsampled 8x)')
    axes[1, 0].axis('off')
    
    # YZ slice (downsampled)
    mid_x = volume.shape[2] // 2
    yz_slice = volume[:, ::8, mid_x]  # Downsample Y by 8x
    axes[1, 1].imshow(yz_slice, cmap='gray', aspect='auto')
    axes[1, 1].set_title(f'YZ slice (x={mid_x}, Y downsampled 8x)')
    axes[1, 1].axis('off')
    
    # Histogram
    # Sample a subset for histogram to avoid memory issues
    sample_data = volume[::4, ::16, ::16].flatten()  # Sample every 4th slice, 16x downsampled
    axes[1, 2].hist(sample_data, bins=50, alpha=0.7)
    axes[1, 2].set_title('Value distribution (sampled)')
    axes[1, 2].set_xlabel('Pixel value')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('stack_check_4k.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Saved 4k stack check visualization to stack_check_4k.pdf")
    
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

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 