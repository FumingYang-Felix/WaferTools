#!/usr/bin/env python3
"""
SIFT Pairwise Alignment Script

Standalone script for aligning and visualizing two high-resolution microscopy sections
using SIFT features, FLANN matching, and RANSAC geometric filtering.

Usage:
    python3 sift_pairwise_alignment.py --folder w7_png_4k --section1 29 --section2 31
    python3 sift_pairwise_alignment.py  # Uses defaults: folder=w7_png_4k, sections 29 and 31

Author: Generated from SIFT analysis pipeline
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
# --- SIFT cache related ---
import pickle, hashlib
from pathlib import Path
from functools import partial 
CACHE_DIR = Path("sift_cache")          # you can change to other fixed path
CACHE_DIR.mkdir(exist_ok=True)


def _prep_job(path, args_dict):
    """single section → generate or skip cache, return friendly information"""
    img = cv2.imread(str(path))
    if img is None:
        return f"✗ {Path(path).name} (read fail)"

    r = args_dict['resize']
    if r != 1.0:
        img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)

    img_pre = texture_rich_color_invariant_preprocessing(img)

    ck = _cache_key(str(path), r,
                    args_dict['sift_features'],
                    args_dict['sift_contrast'],
                    args_dict['sift_edge'])
    if ck.exists():
        return f"✓ {Path(path).name} (cached)"

    sift = cv2.SIFT_create(
        nfeatures=args_dict['sift_features'],
        contrastThreshold=args_dict['sift_contrast'],
        edgeThreshold=args_dict['sift_edge']
    )
    kp, des = sift.detectAndCompute(img_pre, None)
    if kp and des is not None:
        with open(ck, "wb") as f:
            pickle.dump((np.float32([k.pt for k in kp]), des), f)
    return f"✓ {Path(path).name}"



def texture_rich_color_invariant_preprocessing(image):
    """
    Enhanced preprocessing for biological tissue images.
    From the main alignment pipeline - optimized for texture-rich content.
    """
    # Convert to Lab color space for better color invariance
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    
    # Edge enhancement using unsharp masking
    gaussian = cv2.GaussianBlur(enhanced_l, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced_l, 1.5, gaussian, -0.5, 0)
    
    # Ensure values are in valid range
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    
    return unsharp

def _cache_key(img_path: str, resize: float,
               sift_features: int, sift_contrast: float, sift_edge: int) -> Path:
    """
    generate unique md5 → sift_cache/<md5>.pkl
    based on image file content + all SIFT parameters + resize factor
    """
    # ① file content md5 (prevent same name file from being modified)
    with open(img_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    # ② put parameters into key, ensure re-calculation when parameters are changed
    key = f"{md5}_{resize}_{sift_features}_{sift_contrast}_{sift_edge}"
    fname = hashlib.md5(key.encode()).hexdigest() + ".pkl"
    return CACHE_DIR / fname


def perform_sift_alignment(img1, img2, section1_name, section2_name, img1_path, img2_path, current_resize,
                          sift_features=3000, sift_contrast=0.02, sift_edge=20,
                          flann_trees=8, flann_checks=100, lowe_ratio=0.85,
                          ransac_threshold=25.0, min_inlier_ratio=0.08):
    """
    Perform SIFT-based alignment between two images.
    
    Args:
        img1, img2: Input images
        section1_name, section2_name: Section names
        sift_features: Number of SIFT features to detect
        sift_contrast: SIFT contrast threshold
        sift_edge: SIFT edge threshold
        flann_trees: FLANN index trees
        flann_checks: FLANN search checks
        lowe_ratio: Lowe's ratio test threshold
        ransac_threshold: RANSAC reprojection threshold
        min_inlier_ratio: Minimum inlier ratio for success
    
    Returns:
        dict: Results including transformation, matches, inliers, and success status
    """
    print(f"    Detecting SIFT features...")
    start_time = time.time()
    
    # Preprocess images for better feature detection
    processed1 = texture_rich_color_invariant_preprocessing(img1)
    processed2 = texture_rich_color_invariant_preprocessing(img2)
    
    # Initialize SIFT with parameters from UI
    sift = cv2.SIFT_create(
        nfeatures=sift_features,           # From UI
        contrastThreshold=sift_contrast,   # From UI
        edgeThreshold=sift_edge            # From UI
    )
    

    # ------------ SIFT with persistent cache ------------
    def _detect_or_load(img, img_path, resize_factor):
        ck = _cache_key(
            img_path,
            resize=resize_factor,                     # note: here 1.0 means *the img has been resized* in this function
            sift_features=sift_features,
            sift_contrast=sift_contrast,
            sift_edge=sift_edge
        )
        if ck.exists():
            # load in seconds
            with open(ck, "rb") as f:
                kp_xy, des = pickle.load(f)
            kp = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kp_xy]
            return kp, des
        # calculate
        kp, des = sift.detectAndCompute(img, None)
        # only cache when there are keypoints
        if kp and des is not None:
            kp_xy = np.float32([k.pt for k in kp])
            with open(ck, "wb") as f:
                pickle.dump((kp_xy, des), f)
        return kp, des

    kp1, des1 = _detect_or_load(processed1, img1_path, current_resize)
    kp2, des2 = _detect_or_load(processed2, img2_path, current_resize)


    detection_time = time.time() - start_time
    print(f"      Features detected: {len(kp1)} vs {len(kp2)} ({detection_time:.2f}s)")
    
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        print(f"      ✗ Insufficient features detected")
        return None
    
    # FLANN matching with parameters from UI
    print(f"    Matching features with FLANN...")
    match_start = time.time()
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)  # From UI
    search_params = dict(checks=flann_checks)  # From UI
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test with parameter from UI
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowe_ratio * n.distance:  # From UI
                good_matches.append(m)
    
    match_time = time.time() - match_start
    print(f"      FLANN matches: {len(good_matches)} ({match_time:.2f}s)")
    
    if len(good_matches) < 10:
        print(f"      ✗ Insufficient matches: {len(good_matches)}")
        return None
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # RANSAC for rigid transformation with parameters from UI
    print(f"    Applying RANSAC...")
    ransac_start = time.time()
    
    # Use estimateAffinePartial2D for rigid transformation (translation + rotation + uniform scaling)
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,  # From UI
        confidence=0.99,
        maxIters=5000
    )
    
    ransac_time = time.time() - ransac_start
    
    if transformation_matrix is None or inliers is None:
        print(f"      ✗ RANSAC failed")
        return None
    
    num_inliers = np.sum(inliers)
    inlier_ratio = num_inliers / len(good_matches)
    
    print(f"      RANSAC inliers: {num_inliers}/{len(good_matches)} ({inlier_ratio:.1%}) ({ransac_time:.2f}s)")
    
    # Check if alignment is successful with parameter from UI
    if inlier_ratio < min_inlier_ratio:  # From UI
        print(f"      ✗ Insufficient inlier ratio: {inlier_ratio:.1%} < {min_inlier_ratio:.1%}")
        return None
    
    # Extract transformation parameters
    angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]) * 180 / np.pi
    translation_x = transformation_matrix[0, 2]
    translation_y = transformation_matrix[1, 2]
    scale = np.sqrt(transformation_matrix[0, 0]**2 + transformation_matrix[1, 0]**2)
    
    print(f"      ✓ Alignment successful!")
    print(f"        Rotation: {angle:.1f}°")
    print(f"        Translation: ({translation_x:.1f}, {translation_y:.1f})")
    print(f"        Scale: {scale:.3f}")
    
    return {
        'transformation_matrix': transformation_matrix,
        'kp1': kp1, 'kp2': kp2,
        'good_matches': good_matches,
        'inliers': inliers,
        'inlier_ratio': inlier_ratio,
        'num_inliers': num_inliers,
        'angle': angle,
        'translation': (translation_x, translation_y),
        'scale': scale,
        'detection_time': detection_time,
        'match_time': match_time,
        'ransac_time': ransac_time
    }

def calculate_ssim_score(img1, img2, transformation_matrix):
    """Calculate SSIM score on overlapping regions after alignment."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Apply transformation to img1
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (w2, h2))
    
    # Convert to grayscale for SSIM calculation
    if len(img2.shape) == 3:
        gray1 = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = aligned_img1
        gray2 = img2
    
    # ---- Crop to the *true* overlapping area (non-zero pixels in aligned image) ---- #
    mask = gray1 > 0  # Aligned slice has 0-padding outside its field of view

    if mask.sum() < 49:  # Require at least 7×7 valid pixels
        return -1.0, 0
    
    # Bounding box of the overlap
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop1 = gray1[y0:y1, x0:x1]
    crop2 = gray2[y0:y1, x0:x1]

    overlap_pixels = crop1.size

    # Adapt SSIM window size to available region (must be odd and >=3)
    win_est = int(np.sqrt(overlap_pixels))
    win_size = max(3, min(11, (win_est // 2) * 2 - 1))  # odd number between 3 and 11
    
    try:
        ssim_score = ssim(crop1, crop2, win_size=win_size, data_range=255)
        return ssim_score, overlap_pixels
    except Exception as e:
        print(f"        SSIM calculation failed: {e}")
        return -1.0, overlap_pixels

def create_visualization(img1, img2, section1_name, section2_name, results, output_path):
    """Create comprehensive visualization of the alignment results."""
    print(f"    Creating visualization...")
    
    if results is None:
        # Create failure visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'SIFT Alignment: {section1_name} vs {section2_name} - FAILED', fontsize=16, color='red')
        
        axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'{section1_name}')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'{section2_name}')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Apply transformation for overlay
    h2, w2 = img2.shape[:2]
    aligned_img1 = cv2.warpAffine(img1, results['transformation_matrix'], (w2, h2))
    
    # Calculate SSIM
    ssim_score, overlap_pixels = calculate_ssim_score(img1, img2, results['transformation_matrix'])
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 18))
    
    # Title with key metrics
    title = f'SIFT Alignment: {section1_name} vs {section2_name}\n'
    title += f'Features: {len(results["kp1"])} vs {len(results["kp2"])} | '
    title += f'Matches: {len(results["good_matches"])} | '
    title += f'Inliers: {results["num_inliers"]} ({results["inlier_ratio"]:.1%}) | '
    title += f'SSIM: {ssim_score:.3f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Top row: Original images with SIFT matches
    ax1 = plt.subplot(3, 3, 1)
    img_matches = cv2.drawMatches(img1, results['kp1'], img2, results['kp2'], 
                                  results['good_matches'], None, 
                                  matchColor=(0, 255, 0) if results['inliers'] is not None else (255, 0, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Feature Matches\n(Green=Inliers, Red=Outliers)')
    plt.axis('off')
    
    # Draw matches with inlier/outlier distinction
    ax2 = plt.subplot(3, 3, 2)
    # Create custom match visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    
    # Draw match lines
    for i, match in enumerate(results['good_matches']):
        pt1 = tuple(map(int, results['kp1'][match.queryIdx].pt))
        pt2 = tuple(map(int, (results['kp2'][match.trainIdx].pt[0] + w1, results['kp2'][match.trainIdx].pt[1])))
        
        if results['inliers'][i]:
            color = (0, 255, 0)  # Green for inliers
        else:
            color = (255, 0, 0)  # Red for outliers
        
        cv2.line(combined_img, pt1, pt2, color, 1)
    
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title('Match Lines\n(Green=RANSAC Inliers, Red=Outliers)')
    plt.axis('off')
    
    # Transformation info
    ax3 = plt.subplot(3, 3, 3)
    ax3.text(0.1, 0.9, 'Transformation Parameters:', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.8, f'Rotation: {results["angle"]:.2f}°', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.7, f'Translation: ({results["translation"][0]:.1f}, {results["translation"][1]:.1f})', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.6, f'Scale: {results["scale"]:.3f}', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.5, f'SSIM Score: {ssim_score:.3f}', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.4, f'Overlap: {overlap_pixels:,} pixels', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.3, 'Timing:', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.1, 0.2, f'SIFT: {results["detection_time"]:.2f}s', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.1, f'FLANN: {results["match_time"]:.2f}s', fontsize=11, transform=ax3.transAxes)
    ax3.text(0.1, 0.0, f'RANSAC: {results["ransac_time"]:.2f}s', fontsize=11, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Middle row: Before alignment
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title(f'Original {section1_name}')
    plt.axis('off')
    
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title(f'Target {section2_name}')
    plt.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    # Red-Green overlay (before alignment)
    overlay_before = np.zeros_like(img2)
    overlay_before[:, :, 1] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Green channel
    if img1.shape[:2] == img2.shape[:2]:
        overlay_before[:, :, 0] = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Red channel
    plt.imshow(overlay_before)
    plt.title('Before Alignment\n(Red=Sec1, Green=Sec2)')
    plt.axis('off')
    
    # Bottom row: After alignment overlays
    ax7 = plt.subplot(3, 3, 7)
    # Red-Green overlay (after alignment)
    overlay_after = np.zeros_like(img2)
    overlay_after[:, :, 1] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # Green channel
    overlay_after[:, :, 0] = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)  # Red channel
    plt.imshow(overlay_after)
    plt.title('Red-Green Overlay\n(Red=Aligned Sec1, Green=Sec2)')
    plt.axis('off')
    
    ax8 = plt.subplot(3, 3, 8)
    # Alpha blending
    alpha_blend = cv2.addWeighted(aligned_img1, 0.5, img2, 0.5, 0)
    plt.imshow(cv2.cvtColor(alpha_blend, cv2.COLOR_BGR2RGB))
    plt.title('Alpha Blend (50/50)')
    plt.axis('off')
    
    ax9 = plt.subplot(3, 3, 9)
    # Checkerboard pattern
    h, w = img2.shape[:2]
    checkerboard = np.zeros_like(img2)
    square_size = 64  # Size of checkerboard squares
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            # Determine if this square should show img1 or img2
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                # Even squares: show aligned img1
                checkerboard[i:i+square_size, j:j+square_size] = aligned_img1[i:i+square_size, j:j+square_size]
            else:
                # Odd squares: show img2
                checkerboard[i:i+square_size, j:j+square_size] = img2[i:i+square_size, j:j+square_size]
    
    plt.imshow(cv2.cvtColor(checkerboard, cv2.COLOR_BGR2RGB))
    plt.title('Checkerboard Pattern\n(Alternating 64px squares)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      Visualization saved: {output_path}")
    if ssim_score > 0:
        print(f"      SSIM score: {ssim_score:.3f}")
        print(f"      Overlap area: {overlap_pixels:,} pixels")

def find_image_path(folder, section_name):
    for ext in [".png", ".tif"]:
        path = os.path.join(folder, f"{section_name}{ext}")
        if os.path.exists(path):
            return path
    return None

def main():
    parser = argparse.ArgumentParser(description='SIFT Pairwise Alignment for Microscopy Sections')
    parser.add_argument('--folder', type=str, default='w7_png_4k',
                        help='Folder containing section images (default: w7_png_4k)')
    parser.add_argument('--section1', type=int, default=29,
                        help='First section number (default: 29)')
    parser.add_argument('--section2', type=int, default=31,
                        help='Second section number (default: 31)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated)')
    parser.add_argument('--resize', type=float, default=1.0,
                        help='Downscale factor applied to both images before alignment (0<r<=1, 1=no scaling)')
    parser.add_argument('--all_pairs', action='store_true',
                        help='Process every i<j pair of images in --folder')
    parser.add_argument('--out_dir', type=str, default='sift_pairwise_out',
                        help='Directory for CSV & overlay outputs when using --all_pairs')
    parser.add_argument('--cpu_workers', type=int, default=1,
                        help='Parallel workers for --all_pairs (1 = sequential)')
    
    # SIFT Parameters
    parser.add_argument('--sift_features', type=int, default=3000,
                        help='Number of SIFT features to detect (default: 3000)')
    parser.add_argument('--sift_contrast', type=float, default=0.02,
                        help='SIFT contrast threshold (default: 0.02)')
    parser.add_argument('--sift_edge', type=int, default=20,
                        help='SIFT edge threshold (default: 20)')
    
    # FLANN Parameters
    parser.add_argument('--flann_trees', type=int, default=8,
                        help='FLANN index trees (default: 8)')
    parser.add_argument('--flann_checks', type=int, default=100,
                        help='FLANN search checks (default: 100)')
    
    # Matching Parameters
    parser.add_argument('--lowe_ratio', type=float, default=0.85,
                        help='Lowe\'s ratio test threshold (default: 0.85)')
    
    # RANSAC Parameters
    parser.add_argument('--ransac_threshold', type=float, default=25.0,
                        help='RANSAC reprojection threshold (default: 25.0)')
    parser.add_argument('--min_inlier_ratio', type=float, default=0.08,
                        help='Minimum inlier ratio for success (default: 0.08)')
    
    parser.add_argument('--ssim_scale_min', type=float, default=0.9,
                        help='Only compute SSIM when scale >= this value')
    parser.add_argument('--ssim_scale_max', type=float, default=1.1,
                        help='Only compute SSIM when scale <= this value')
    
    args = parser.parse_args()
    
    if args.all_pairs:
        run_all_pairs(args)
        return

    # add this line after parsing args in main() (for later reference)
    current_resize = args.resize

    # ---------------- single-pair legacy mode ---------------- # 
    section1_name = f"section_{args.section1}_r01_c01"
    section2_name = f"section_{args.section2}_r01_c01"
    img1_path = find_image_path(args.folder, section1_name)
    img2_path = find_image_path(args.folder, section2_name)
    if img1_path is None or img2_path is None:
        print(f"Error: Could not find image for one or both sections")
        return
    
    # Load images
    print(f"Loading images...")
    print(f"  Section 1: {img1_path}")
    print(f"  Section 2: {img2_path}")
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images")
        return
    
    if args.resize <= 0 or args.resize > 1:
        print(f"Error: --resize must be in (0,1]; got {args.resize}")
        return

    if args.resize != 1.0:
        img1 = cv2.resize(img1, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        print(f"  Images downscaled by {args.resize:g} → {img1.shape} vs {img2.shape}")
    
    print(f"  Image dimensions: {img1.shape} vs {img2.shape}")
    
    # Perform alignment
    print(f"\nPerforming SIFT alignment...")
    total_start = time.time()
    
    results = perform_sift_alignment(
        img1, img2, section1_name, section2_name,
        img1_path, img2_path,    
        current_resize,  
        sift_features=args.sift_features,
        sift_contrast=args.sift_contrast,
        sift_edge=args.sift_edge,
        flann_trees=args.flann_trees,
        flann_checks=args.flann_checks,
        lowe_ratio=args.lowe_ratio,
        ransac_threshold=args.ransac_threshold,
        min_inlier_ratio=args.min_inlier_ratio
    )
    
    total_time = time.time() - total_start
    print(f"  Total alignment time: {total_time:.2f}s")
    
    # Generate output filename
    if args.output is None:
        output_path = f"sift_alignment_{section1_name}_vs_{section2_name}.png"
    else:
        output_path = args.output
    
    # Create visualization
    print(f"\nCreating visualization...")
    # create_visualization(img1, img2, section1_name, section2_name, results, output_path)
    
    # Summary
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"Sections: {args.section1} vs {args.section2}")
    if results is not None:
        print(f"Status: ✓ SUCCESS")
        print(f"Inlier ratio: {results['inlier_ratio']:.1%}")
        print(f"Transformation: {results['angle']:.1f}° rotation, ({results['translation'][0]:.1f}, {results['translation'][1]:.1f}) translation")
    else:
        print(f"Status: ✗ FAILED")
    print(f"Visualization: {output_path}")
    print(f"Total time: {total_time:.2f}s")

# --------------- multi-pair helpers -------------------------- #

def _pair_job(job):
    path_a, path_b, args_dict = job
    args = argparse.Namespace(**args_dict)

    img1 = cv2.imread(path_a)
    img2 = cv2.imread(path_b)

    if img1 is None or img2 is None:
        print(f"failed to read {path_a} or {path_b}")
        return None

    # optional resize
    if args.resize != 1.0:
        img1 = cv2.resize(img1, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    name1 = Path(path_a).stem
    name2 = Path(path_b).stem

    # Pass SIFT parameters to perform_sift_alignment
    results = perform_sift_alignment(
        img1, img2, name1, name2,
        path_a,          # img1_path  ← the real file path
        path_b,          # img2_path
        args.resize,     # current_resize
        sift_features=getattr(args, 'sift_features', 3000),
        sift_contrast=getattr(args, 'sift_contrast', 0.02),
        sift_edge=getattr(args, 'sift_edge', 20),
        flann_trees=getattr(args, 'flann_trees', 8),
        flann_checks=getattr(args, 'flann_checks', 100),
        lowe_ratio=getattr(args, 'lowe_ratio', 0.85),
        ransac_threshold=getattr(args, 'ransac_threshold', 25.0),
        min_inlier_ratio=getattr(args, 'min_inlier_ratio', 0.08)
    )

    # prepare overlay regardless of success
    # overlay_path = Path(args.out_dir) / f"{name1}_vs_{name2}.png"
    # try:
    #     create_visualization(img1, img2, name1, name2, results, str(overlay_path))
    # except Exception as e:
    #     print(f"Overlay failed for {name1},{name2}: {e}")

    if results is None:
        return (name1, name2, None, None, None, None, 0.0, 0, -1.0, 0)

    dx, dy = results['translation']
    if args.resize != 1.0:
        dx /= args.resize
        dy /= args.resize

    # SSIM / overlap
    if (args.ssim_scale_min <= results['scale'] <= args.ssim_scale_max):
        ssim_score, overlap_px = calculate_ssim_score(
            img1, img2, results['transformation_matrix'])
    else:
        ssim_score, overlap_px = -1.0, 0     # 记录无效 / 跳过


    return (name1, name2, dx, dy, results['angle'], results['scale'],
            results['inlier_ratio'], results['num_inliers'], ssim_score, overlap_px)

def run_all_pairs(args):
    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    # ------------------------------------------------------------
    # collect all slice paths
    # ------------------------------------------------------------
    img_paths = sorted(list(folder.glob("*.png")) + list(folder.glob("*.tif")))
    if len(img_paths) < 2:
        print("Need at least two PNG or TIF images for --all_pairs mode")
        return

    # ------------------------------------------------------------
    # A. parallel pre-compute / read SIFT cache
    # ------------------------------------------------------------
    print(f"Pre-computing SIFT cache for {len(img_paths)} sections …")

    prep_arg = dict(
        resize=args.resize,
        sift_features=args.sift_features,
        sift_contrast=args.sift_contrast,
        sift_edge=args.sift_edge,
    )
    prep_worker = partial(_prep_job, args_dict=prep_arg)

    with ProcessPoolExecutor(max_workers=args.cpu_workers or 4) as pool:
        for msg in tqdm(pool.map(prep_worker, img_paths),
                        total=len(img_paths), desc="cache"):
            print(msg)

    print("🎉  Cache build finished.\n")

    # ------------------------------------------------------------
    # B. formal pair-wise alignment
    # ------------------------------------------------------------
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)   # add this line!
    # overlay_dir = out_dir / "overlays"
    # overlay_dir.mkdir(parents=True, exist_ok=True)

    arg_dict = {
        'resize'          : args.resize,
        'out_dir'         : str(out_dir),
        'sift_features'   : args.sift_features,
        'sift_contrast'   : args.sift_contrast,
        'sift_edge'       : args.sift_edge,
        'flann_trees'     : args.flann_trees,
        'flann_checks'    : args.flann_checks,
        'lowe_ratio'      : args.lowe_ratio,
        'ransac_threshold': args.ransac_threshold,
        'min_inlier_ratio': args.min_inlier_ratio,
        'ssim_scale_min'  : args.ssim_scale_min,
        'ssim_scale_max'  : args.ssim_scale_max,
    }

    jobs = [(str(img_paths[i]), str(img_paths[j]), arg_dict)
            for i in range(len(img_paths) - 1)
            for j in range(i + 1, len(img_paths))]

    print(f"Processing {len(jobs)} pairs from {len(img_paths)} images …")

    records = []
    if args.cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            futures = [pool.submit(_pair_job, jb) for jb in jobs]
            for fut in tqdm(as_completed(futures), total=len(jobs), desc="pairs"):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
    else:
        for jb in tqdm(jobs, desc="pairs"):
            rec = _pair_job(jb)
            if rec is not None:
                records.append(rec)

    if not records:
        print("No successful alignments.")
        return

    df = pd.DataFrame(records, columns=[
        'fixed', 'moving', 'dx_px', 'dy_px', 'angle_deg', 'scale',
        'inlier_ratio', 'num_inliers', 'ssim', 'overlap_px'
    ])

    csv_path = out_dir / "pairwise_alignment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV written to {csv_path}")
    #print(f"Overlays saved to {overlay_dir}")




if __name__ == "__main__":
    main() 
