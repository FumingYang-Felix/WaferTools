#!/usr/bin/env python3
"""
create_best_pairs_pdf.py - Create PDF showing each section with its best pair
Uses num_inliers × SSIM as combined score
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from PIL import Image
import os

def load_pair_scores_grouped(csv_path: Path):
    """Load pairwise similarity CSV, group by 'fixed', and compute combined score = num_inliers × SSIM × SSIM. 不做任何过滤。"""
    groups = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 9:
                try:
                    fixed = row[0].strip()
                    moving = row[1].strip()
                    num_inliers = float(row[7])
                    ssim = float(row[8])
                    combined = num_inliers * ssim * ssim
                    if fixed not in groups:
                        groups[fixed] = []
                    groups[fixed].append((moving, combined, num_inliers, ssim))
                except (ValueError, IndexError):
                    continue
    return groups

def find_best_pairs_grouped(groups: dict):
    """For each section (fixed), pick the moving with max combined score."""
    best_pairs = {}
    for fixed, lst in groups.items():
        if not lst:
            continue
        best = max(lst, key=lambda x: x[1])
        best_pairs[fixed] = best  # (moving, combined, num_inliers, ssim)
    return best_pairs

def load_image(image_path: Path, target_size=(200, 200)):
    """Load and resize image."""
    try:
        img = Image.open(image_path)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        # Return blank image
        return np.zeros(target_size, dtype=np.uint8)

def create_best_pairs_pdf_grouped(best_pairs: dict, image_dir: Path, output_pdf: str, images_per_page=4, target_size=(200, 200)):
    """Create PDF showing each section with its best pair."""
    from matplotlib.backends.backend_pdf import PdfPages
    sections = list(best_pairs.keys())
    # 按分数排序
    sorted_sections = sorted(sections, key=lambda s: best_pairs[s][1], reverse=True)
    with PdfPages(output_pdf) as pdf:
        page_num = 0
        for i in range(0, len(sorted_sections), images_per_page):
            page_sections = sorted_sections[i:i+images_per_page]
            fig, axes = plt.subplots(images_per_page, 3, figsize=(15, 5*images_per_page))
            if images_per_page == 1:
                axes = axes.reshape(1, -1)
            for j, section in enumerate(page_sections):
                moving, combined, num_inliers, ssim = best_pairs[section]
                img1_path = image_dir / f"{section}.png"
                img2_path = image_dir / f"{moving}.png"
                img1 = load_image(img1_path, target_size)
                img2 = load_image(img2_path, target_size)
                axes[j, 0].imshow(img1, cmap='gray')
                axes[j, 0].set_title(f'{section}', fontsize=10)
                axes[j, 0].axis('off')
                axes[j, 1].imshow(img2, cmap='gray')
                axes[j, 1].set_title(f'{moving}\n(Best Pair)', fontsize=10)
                axes[j, 1].axis('off')
                axes[j, 2].text(0.1, 0.8, f'Section: {section}', fontsize=12, fontweight='bold')
                axes[j, 2].text(0.1, 0.6, f'Best Pair: {moving}', fontsize=10)
                axes[j, 2].text(0.1, 0.5, f'Combined: {combined:.1f}', fontsize=10)
                axes[j, 2].text(0.1, 0.4, f'num_inliers: {num_inliers:.1f}', fontsize=10)
                axes[j, 2].text(0.1, 0.3, f'ssim: {ssim:.4f}', fontsize=10)
                axes[j, 2].text(0.1, 0.2, f'Rank: {i+j+1}', fontsize=10)
                axes[j, 2].axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1
            print(f"Created page {page_num} with sections {i+1}-{min(i+images_per_page, len(sections))}")
    print(f"PDF saved to {output_pdf}")

def create_all_pairs_list_grouped(best_pairs: dict, output_file: str):
    """Create a text file listing all best pairs."""
    sorted_sections = sorted(best_pairs.keys(), key=lambda s: best_pairs[s][1], reverse=True)
    with open(output_file, 'w') as f:
        f.write("All 106 sections with their best pairs (single direction, grouped by fixed)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Format: Section (fixed) -> Best Pair (moving) (Combined Score)\n\n")
        for i, section in enumerate(sorted_sections):
            moving, combined, num_inliers, ssim = best_pairs[section]
            f.write(f"{i+1:3d}. {section} -> {moving} (score: {combined:.1f}, inliers: {num_inliers:.1f}, ssim: {ssim:.4f})\n")
    print(f"All pairs list saved to {output_file}")

def main():
    ap = argparse.ArgumentParser(description='Create PDF showing sections with their best pairs (grouped by fixed)')
    ap.add_argument('pair_csv', help='CSV with pairwise scores')
    ap.add_argument('image_dir', help='Directory containing section images')
    ap.add_argument('--output', default='best_pairs_grouped.pdf', help='Output PDF file')
    ap.add_argument('--list', default='all_pairs_grouped.txt', help='Text file with all pairs')
    ap.add_argument('--images-per-page', type=int, default=4, help='Images per page')
    args = ap.parse_args()
    groups = load_pair_scores_grouped(Path(args.pair_csv))
    best_pairs = find_best_pairs_grouped(groups)
    print(f"Loaded {len(best_pairs)} sections with best pairs (grouped by fixed)")
    create_best_pairs_pdf_grouped(best_pairs, Path(args.image_dir), args.output, args.images_per_page)
    create_all_pairs_list_grouped(best_pairs, args.list)

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 