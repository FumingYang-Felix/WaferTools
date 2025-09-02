#!/usr/bin/env python3
"""
SIFT Pairwise Alignment Script (Single Pair: section_6_r01_c01 vs section_1_r01_c01)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from skimage.metrics import structural_similarity as ssim

def texture_rich_color_invariant_preprocessing(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    gaussian = cv2.GaussianBlur(enhanced_l, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced_l, 1.5, gaussian, -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    return unsharp

def perform_sift_alignment(img1, img2, section1_name, section2_name, 
                          sift_features=3000, sift_contrast=0.02, sift_edge=20,
                          flann_trees=8, flann_checks=100, lowe_ratio=0.85,
                          ransac_threshold=25.0, min_inlier_ratio=0.08):
    print(f"    Detecting SIFT features...")
    import time
    start_time = time.time()
    processed1 = texture_rich_color_invariant_preprocessing(img1)
    processed2 = texture_rich_color_invariant_preprocessing(img2)
    sift = cv2.SIFT_create(
        nfeatures=sift_features,
        contrastThreshold=sift_contrast,
        edgeThreshold=sift_edge
    )
    kp1, des1 = sift.detectAndCompute(processed1, None)
    kp2, des2 = sift.detectAndCompute(processed2, None)
    detection_time = time.time() - start_time
    print(f"      Features detected: {len(kp1)} vs {len(kp2)} ({detection_time:.2f}s)")
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        print(f"      ✗ Insufficient features detected")
        return None
    print(f"    Matching features with FLANN...")
    match_start = time.time()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees)
    search_params = dict(checks=flann_checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowe_ratio * n.distance:
                good_matches.append(m)
    match_time = time.time() - match_start
    print(f"      FLANN matches: {len(good_matches)} ({match_time:.2f}s)")
    if len(good_matches) < 10:
        print(f"      ✗ Insufficient matches: {len(good_matches)}")
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    print(f"    Applying RANSAC...")
    ransac_start = time.time()
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
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
    if inlier_ratio < min_inlier_ratio:
        print(f"      ✗ Insufficient inlier ratio: {inlier_ratio:.1%} < {min_inlier_ratio:.1%}")
        return None
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
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (w2, h2))
    if len(img2.shape) == 3:
        gray1 = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = aligned_img1
        gray2 = img2
    mask = gray1 > 0
    if mask.sum() < 49:
        return -1.0, 0
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop1 = gray1[y0:y1, x0:x1]
    crop2 = gray2[y0:y1, x0:x1]
    overlap_pixels = crop1.size
    win_est = int(np.sqrt(overlap_pixels))
    win_size = max(3, min(11, (win_est // 2) * 2 - 1))
    try:
        ssim_score = ssim(crop1, crop2, win_size=win_size, data_range=255)
        return ssim_score, overlap_pixels
    except Exception as e:
        print(f"        SSIM calculation failed: {e}")
        return -1.0, overlap_pixels

def create_visualization(img1, img2, section1_name, section2_name, results, output_path):
    print(f"    Creating visualization...")
    if results is None:
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
    h2, w2 = img2.shape[:2]
    aligned_img1 = cv2.warpAffine(img1, results['transformation_matrix'], (w2, h2))
    ssim_score, overlap_pixels = calculate_ssim_score(img1, img2, results['transformation_matrix'])
    fig = plt.figure(figsize=(24, 18))
    title = f'SIFT Alignment: {section1_name} vs {section2_name}\n'
    title += f'Features: {len(results["kp1"])} vs {len(results["kp2"])} | '
    title += f'Matches: {len(results["good_matches"])} | '
    title += f'Inliers: {results["num_inliers"]} ({results["inlier_ratio"]:.1%}) | '
    title += f'SSIM: {ssim_score:.3f}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    ax1 = plt.subplot(3, 3, 1)
    img_matches = cv2.drawMatches(img1, results['kp1'], img2, results['kp2'], 
                                  results['good_matches'], None, 
                                  matchColor=(0, 255, 0) if results['inliers'] is not None else (255, 0, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Feature Matches\n(Green=Inliers, Red=Outliers)')
    plt.axis('off')
    ax2 = plt.subplot(3, 3, 2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    for i, match in enumerate(results['good_matches']):
        pt1 = tuple(map(int, results['kp1'][match.queryIdx].pt))
        pt2 = tuple(map(int, (results['kp2'][match.trainIdx].pt[0] + w1, results['kp2'][match.trainIdx].pt[1])))
        if results['inliers'][i]:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.line(combined_img, pt1, pt2, color, 1)
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.title('Match Lines\n(Green=RANSAC Inliers, Red=Outliers)')
    plt.axis('off')
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
    ax4 = plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title(f'Original {section1_name}')
    plt.axis('off')
    ax5 = plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title(f'Target {section2_name}')
    plt.axis('off')
    ax6 = plt.subplot(3, 3, 6)
    overlay_before = np.zeros_like(img2)
    overlay_before[:, :, 1] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if img1.shape[:2] == img2.shape[:2]:
        overlay_before[:, :, 0] = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    plt.imshow(overlay_before)
    plt.title('Before Alignment\n(Red=Sec1, Green=Sec2)')
    plt.axis('off')
    ax7 = plt.subplot(3, 3, 7)
    overlay_after = np.zeros_like(img2)
    overlay_after[:, :, 1] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    overlay_after[:, :, 0] = cv2.cvtColor(aligned_img1, cv2.COLOR_BGR2GRAY)
    plt.imshow(overlay_after)
    plt.title('Red-Green Overlay\n(Red=Aligned Sec1, Green=Sec2)')
    plt.axis('off')
    ax8 = plt.subplot(3, 3, 8)
    alpha_blend = cv2.addWeighted(aligned_img1, 0.5, img2, 0.5, 0)
    plt.imshow(cv2.cvtColor(alpha_blend, cv2.COLOR_BGR2RGB))
    plt.title('Alpha Blend (50/50)')
    plt.axis('off')
    ax9 = plt.subplot(3, 3, 9)
    h, w = img2.shape[:2]
    checkerboard = np.zeros_like(img2)
    square_size = 64
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = aligned_img1[i:i+square_size, j:j+square_size]
            else:
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
    parser.add_argument('--folder', type=str, default='images_png', help='Folder containing section images')
    parser.add_argument('--resize', type=float, default=0.15, help='Downscale factor applied to both images before alignment (0<r<=1, 1=no scaling)')
    args = parser.parse_args()
    section1_name = "section_6_r01_c01"
    section2_name = "section_1_r01_c01"
    img1_path = find_image_path(args.folder, section1_name)
    img2_path = find_image_path(args.folder, section2_name)
    if img1_path is None or img2_path is None:
        print(f"Error: Could not find image for one or both sections")
        return
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
    print(f"\nPerforming SIFT alignment...")
    import time
    total_start = time.time()
    results = perform_sift_alignment(
        img1, img2, section1_name, section2_name,
        sift_features=3000,
        sift_contrast=0.02,
        sift_edge=20,
        flann_trees=8,
        flann_checks=100,
        lowe_ratio=0.85,
        ransac_threshold=25.0,
        min_inlier_ratio=0.08
    )
    total_time = time.time() - total_start
    print(f"  Total alignment time: {total_time:.2f}s")
    output_path = f"sift_alignment_{section1_name}_vs_{section2_name}.png"
    print(f"\nCreating visualization...")
    create_visualization(img1, img2, section1_name, section2_name, results, output_path)
    print(f"\n=== ALIGNMENT SUMMARY ===")
    print(f"Sections: {section1_name} vs {section2_name}")
    if results is not None:
        print(f"Status: ✓ SUCCESS")
        print(f"Inlier ratio: {results['inlier_ratio']:.1%}")
        print(f"Transformation: {results['angle']:.1f}° rotation, ({results['translation'][0]:.1f}, {results['translation'][1]:.1f}) translation")
    else:
        print(f"Status: ✗ FAILED")
    print(f"Visualization: {output_path}")
    print(f"Total time: {total_time:.2f}s")

def draw_global_match_lines(folder, chain_file, resize=0.15, output_path='global_match_lines.png', overlay_output_path='global_match_lines_with_overlay.png'):
    # 1. Parse final order from chain.txt
    def parse_chain_file(chain_file_path):
        with open(chain_file_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if 'final single chain' in line and ':' in line:
                chain_part = line.split(':')[1].strip()
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        for line in lines:
            if 'chain' in line and '->' in line and 'section_' in line:
                chain_part = line.split(':')[1].strip() if ':' in line else line
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        return None

    section_order = parse_chain_file(chain_file)
    if not section_order:
        print('Error: Could not parse section order from chain file')
        return
    print(f'Final order: {section_order}')

    # 2. Load and resize all images
    images = []
    for sec in section_order:
        for ext in ['.png', '.tif']:
            img_path = os.path.join(folder, f'{sec}{ext}')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    images.append(img)
                    break
        else:
            print(f'Image not found for {sec}')
            return
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    canvas_height = max(heights)
    # offsets只记录每个section左上角x坐标
    gap_px = 100
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1] + w + gap_px)
    canvas_width = sum(widths) + gap_px * (len(widths) - 1)
    # 主图
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[:h, offsets[i]:offsets[i]+w] = img
    # label行
    label_height = 150
    label_img = np.ones((label_height, canvas_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    font_thickness = 8
    for i, sec in enumerate(section_order):
        w = widths[i]
        x0 = offsets[i]
        x1 = x0 + w
        text = sec
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x0 + (w - text_w) // 2
        text_y = label_height // 2 + text_h // 2
        cv2.putText(label_img, text, (text_x, text_y), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    # overlays和match lines
    overlays = []
    overlay_heights = []
    for i in range(len(images)-1):
        img1 = images[i]
        img2 = images[i+1]
        processed1 = texture_rich_color_invariant_preprocessing(img1)
        processed2 = texture_rich_color_invariant_preprocessing(img2)
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=20)
        kp1, des1 = sift.detectAndCompute(processed1, None)
        kp2, des2 = sift.detectAndCompute(processed2, None)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            continue
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.85 * n.distance:
                    good_matches.append(m)
        if len(good_matches) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0, confidence=0.99, maxIters=5000)
        if M is None or inliers is None:
            overlays.append(None)
            overlay_heights.append(0)
            continue
        # 画绿色match lines在canvas上（pt1=offsets[i]，pt2=offsets[i+1]）
        for match, inlier in zip(good_matches, inliers.flatten()):
            if inlier:
                pt1 = np.array(kp1[match.queryIdx].pt) + [offsets[i], 0]
                pt2 = np.array(kp2[match.trainIdx].pt) + [offsets[i+1], 0]
                cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
        # 生成overlay，居中于gap
        h2, w2 = img2.shape[:2]
        aligned_img1 = cv2.warpAffine(img1, M, (w2, h2))
        overlay = np.zeros((h2, w2, 3), dtype=np.uint8)
        overlay[..., 1] = img2[..., 1]
        overlay[..., 2] = aligned_img1[..., 2]
        mask = (aligned_img1.sum(axis=2) > 0)
        overlay[mask, 1] = img2[mask, 1]
        overlay[mask, 2] = aligned_img1[mask, 2]
        overlays.append(overlay)
        overlay_heights.append(h2)
    # overlay行
    max_overlay_height = max(overlay_heights) if overlay_heights else 0
    overlay_row = np.ones((max_overlay_height, canvas_width, 3), dtype=np.uint8) * 40
    for i, overlay in enumerate(overlays):
        if overlay is not None:
            h, w = overlay.shape[:2]
            # overlay严格居中于gap
            gap_left = offsets[i] + widths[i]
            gap_right = offsets[i+1]
            gap_center = (gap_left + gap_right) // 2
            x0 = max(gap_center - w // 2, 0)
            x1 = min(x0 + w, canvas_width)
            if x1 - x0 < w:
                overlay = overlay[:, :x1-x0]
            overlay_row[:h, x0:x1] = overlay
    # 合并label、主图和overlay
    gap = 20
    total_height = label_height + canvas_height + gap + max_overlay_height
    final_img = np.ones((total_height, canvas_width, 3), dtype=np.uint8) * 40
    final_img[:label_height, :, :] = label_img
    final_img[label_height:label_height+canvas_height, :, :] = canvas
    final_img[label_height+canvas_height+gap:label_height+canvas_height+gap+max_overlay_height, :, :] = overlay_row
    cv2.imwrite(output_path, canvas)
    cv2.imwrite(overlay_output_path, final_img)
    print(f'Global match lines image saved to {output_path}')
    print(f'Global match lines with overlay image saved to {overlay_output_path}')

def draw_global_match_lines_s_shape(folder, chain_file, resize=0.15, output_path='global_match_lines_s.png', overlay_output_path='global_match_lines_s_with_overlay.png', max_sections_per_row=8, gap_px=100):
    # 1. Parse final order from chain.txt
    def parse_chain_file(chain_file_path):
        with open(chain_file_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if 'final single chain' in line and ':' in line:
                chain_part = line.split(':')[1].strip()
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        for line in lines:
            if 'chain' in line and '->' in line and 'section_' in line:
                chain_part = line.split(':')[1].strip() if ':' in line else line
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        return None

    section_order = parse_chain_file(chain_file)
    if not section_order:
        print('Error: Could not parse section order from chain file')
        return
    print(f'Final order: {section_order}')

    # 2. Load and resize all images
    images = []
    for sec in section_order:
        for ext in ['.png', '.tif']:
            img_path = os.path.join(folder, f'{sec}{ext}')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    images.append(img)
                    break
        else:
            print(f'Image not found for {sec}')
            return
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    n_sections = len(images)
    # 3. 分行分配section，S形排列
    rows = []
    idx = 0
    while idx < n_sections:
        row = list(range(idx, min(idx+max_sections_per_row, n_sections)))
        rows.append(row)
        idx += max_sections_per_row
    # S形：偶数行正序，奇数行逆序
    for i, row in enumerate(rows):
        if i % 2 == 1:
            row.reverse()
    # 4. 逐行拼接主图和overlay
    label_height = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    font_thickness = 8
    row_canvases = []
    row_labels = []
    row_overlays = []
    row_overlay_heights = []
    for row_idx, row in enumerate(rows):
        row_imgs = [images[i] for i in row]
        row_widths = [widths[i] for i in row]
        row_heights = [heights[i] for i in row]
        row_offsets = [0]
        for w in row_widths[:-1]:
            row_offsets.append(row_offsets[-1] + w + gap_px)
        row_canvas_width = sum(row_widths) + gap_px * (len(row_widths)-1)
        row_canvas_height = max(row_heights)
        row_canvas = np.ones((row_canvas_height, row_canvas_width, 3), dtype=np.uint8) * 255
        for i, img in enumerate(row_imgs):
            h, w = img.shape[:2]
            row_canvas[:h, row_offsets[i]:row_offsets[i]+w] = img
        # label
        label_img = np.ones((label_height, row_canvas_width, 3), dtype=np.uint8) * 255
        for i, sec_idx in enumerate(row):
            w = widths[sec_idx]
            x0 = row_offsets[i]
            text = section_order[sec_idx]
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x0 + (w - text_w) // 2
            text_y = label_height // 2 + text_h // 2
            cv2.putText(label_img, text, (text_x, text_y), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
        # overlay
        overlays = []
        overlay_heights = []
        for i in range(len(row)-1):
            idx1 = row[i]
            idx2 = row[i+1]
            img1 = images[idx1]
            img2 = images[idx2]
            processed1 = texture_rich_color_invariant_preprocessing(img1)
            processed2 = texture_rich_color_invariant_preprocessing(img2)
            sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=20)
            kp1, des1 = sift.detectAndCompute(processed1, None)
            kp2, des2 = sift.detectAndCompute(processed2, None)
            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                overlays.append(None)
                overlay_heights.append(0)
                continue
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.85 * n.distance:
                        good_matches.append(m)
            if len(good_matches) < 10:
                overlays.append(None)
                overlay_heights.append(0)
                continue
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0, confidence=0.99, maxIters=5000)
            if M is None or inliers is None:
                overlays.append(None)
                overlay_heights.append(0)
                continue
            # 画绿色match lines在row_canvas上
            for match, inlier in zip(good_matches, inliers.flatten()):
                if inlier:
                    pt1 = np.array(kp1[match.queryIdx].pt) + [row_offsets[i], 0]
                    pt2 = np.array(kp2[match.trainIdx].pt) + [row_offsets[i+1], 0]
                    cv2.line(row_canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
            # 生成overlay，居中于gap
            h2, w2 = img2.shape[:2]
            aligned_img1 = cv2.warpAffine(img1, M, (w2, h2))
            overlay = np.zeros((h2, w2, 3), dtype=np.uint8)
            overlay[..., 1] = img2[..., 1]
            overlay[..., 2] = aligned_img1[..., 2]
            mask = (aligned_img1.sum(axis=2) > 0)
            overlay[mask, 1] = img2[mask, 1]
            overlay[mask, 2] = aligned_img1[mask, 2]
            overlays.append(overlay)
            overlay_heights.append(h2)
        # overlay行
        max_overlay_height = max(overlay_heights) if overlay_heights else 0
        overlay_row = np.ones((max_overlay_height, row_canvas_width, 3), dtype=np.uint8) * 40
        for i, overlay in enumerate(overlays):
            if overlay is not None:
                h, w = overlay.shape[:2]
                gap_left = row_offsets[i] + row_widths[i]
                gap_right = row_offsets[i+1]
                gap_center = (gap_left + gap_right) // 2
                x0 = max(gap_center - w // 2, 0)
                x1 = min(x0 + w, row_canvas_width)
                if x1 - x0 < w:
                    overlay = overlay[:, :x1-x0]
                overlay_row[:h, x0:x1] = overlay
        row_canvases.append(row_canvas)
        row_labels.append(label_img)
        row_overlays.append(overlay_row)
        row_overlay_heights.append(max_overlay_height)
    # 5. 竖直拼接所有行
    gap_row = 40
    total_height = 0
    total_width = max([c.shape[1] for c in row_canvases])
    for i in range(len(row_canvases)):
        total_height += label_height + row_canvases[i].shape[0] + gap_row + row_overlay_heights[i]
    final_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 40
    y = 0
    for i in range(len(row_canvases)):
        w = row_canvases[i].shape[1]
        # 居中对齐
        x = (total_width - w) // 2
        final_img[y:y+label_height, x:x+w] = row_labels[i]
        y += label_height
        final_img[y:y+row_canvases[i].shape[0], x:x+w] = row_canvases[i]
        y += row_canvases[i].shape[0]
        y += gap_row
        final_img[y:y+row_overlay_heights[i], x:x+w] = row_overlays[i]
        y += row_overlay_heights[i]
    cv2.imwrite(output_path, final_img)
    print(f'S-shape global match lines with overlay image saved to {output_path}')

def draw_global_match_lines_with_overlap_arrows(folder, chain_file, resize=0.15, output_path='global_match_lines_with_overlap_arrows.png', gap_px=100):
    # 1. Parse final order from chain.txt
    def parse_chain_file(chain_file_path):
        with open(chain_file_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if 'final single chain' in line and ':' in line:
                chain_part = line.split(':')[1].strip()
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        for line in lines:
            if 'chain' in line and '->' in line and 'section_' in line:
                chain_part = line.split(':')[1].strip() if ':' in line else line
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        return None

    section_order = parse_chain_file(chain_file)
    if not section_order:
        print('Error: Could not parse section order from chain file')
        return
    print(f'Final order: {section_order}')

    # 2. Load and resize all images
    images = []
    for sec in section_order:
        for ext in ['.png', '.tif']:
            img_path = os.path.join(folder, f'{sec}{ext}')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    images.append(img)
                    break
        else:
            print(f'Image not found for {sec}')
            return
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    n_sections = len(images)
    # offsets
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1] + w + gap_px)
    canvas_width = sum(widths) + gap_px * (len(widths) - 1)
    label_height = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    font_thickness = 8
    # 第一行：section label
    label_img = np.ones((label_height, canvas_width, 3), dtype=np.uint8) * 255
    for i, sec in enumerate(section_order):
        w = widths[i]
        x0 = offsets[i]
        text = sec
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x0 + (w - text_w) // 2
        text_y = label_height // 2 + text_h // 2
        cv2.putText(label_img, text, (text_x, text_y), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    # 第一行：主图
    canvas_height = max(heights)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[:h, offsets[i]:offsets[i]+w] = img
    # 第二行：overlay
    overlays = []
    overlay_heights = []
    overlap_imgs = []
    overlap_heights = []
    for i in range(n_sections-1):
        img1 = images[i]
        img2 = images[i+1]
        processed1 = texture_rich_color_invariant_preprocessing(img1)
        processed2 = texture_rich_color_invariant_preprocessing(img2)
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=20)
        kp1, des1 = sift.detectAndCompute(processed1, None)
        kp2, des2 = sift.detectAndCompute(processed2, None)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            continue
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.85 * n.distance:
                    good_matches.append(m)
        if len(good_matches) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0, confidence=0.99, maxIters=5000)
        if M is None or inliers is None:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            continue
        # 画绿色match lines在canvas上
        for match, inlier in zip(good_matches, inliers.flatten()):
            if inlier:
                pt1 = np.array(kp1[match.queryIdx].pt) + [offsets[i], 0]
                pt2 = np.array(kp2[match.trainIdx].pt) + [offsets[i+1], 0]
                cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
        # 生成overlay
        h2, w2 = img2.shape[:2]
        aligned_img1 = cv2.warpAffine(img1, M, (w2, h2))
        overlay = np.zeros((h2, w2, 3), dtype=np.uint8)
        overlay[..., 1] = img2[..., 1]
        overlay[..., 2] = aligned_img1[..., 2]
        mask = (aligned_img1.sum(axis=2) > 0)
        overlay[mask, 1] = img2[mask, 1]
        overlay[mask, 2] = aligned_img1[mask, 2]
        overlays.append(overlay)
        overlay_heights.append(h2)
        # 生成overlap alpha blend（只保留overlap区域）
        overlap_mask = (aligned_img1.sum(axis=2) > 0) & (img2.sum(axis=2) > 0)
        alpha_blend = cv2.addWeighted(aligned_img1, 0.5, img2, 0.5, 0)
        overlap_img = np.zeros_like(img2)
        overlap_img[overlap_mask] = alpha_blend[overlap_mask]
        overlap_imgs.append(overlap_img)
        overlap_heights.append(h2)
    # overlay行和overlap行都只画n-1个块，居中在gap中心，严格上下对齐
    overlay_row = np.ones((max_overlay_height, canvas_width, 3), dtype=np.uint8) * 40
    overlap_row = np.ones((max_overlap_height, canvas_width, 3), dtype=np.uint8) * 40
    for i in range(n_sections-1):
        # overlay
        overlay = overlays[i]
        if overlay is not None:
            h, w = overlay.shape[:2]
            x_center = gap_centers[i]
            x0 = max(0, x_center - w//2)
            x1 = min(canvas_width, x0 + w)
            if x1 - x0 < w:
                overlay = overlay[:, :x1-x0]
            y0 = (max_overlay_height - h) // 2
            overlay_row[y0:y0+h, x0:x1] = overlay
            cv2.rectangle(overlay_row, (x0, y0), (x1-1, y0+h-1), (255,255,255), 4)
        # overlap
        overlap_img = overlap_imgs[i]
        if overlap_img is not None:
            h, w = overlap_img.shape[:2]
            x_center = gap_centers[i]
            x0 = max(0, x_center - w//2)
            x1 = min(canvas_width, x0 + w)
            if x1 - x0 < w:
                overlap_img = overlap_img[:, :x1-x0]
            y0 = (max_overlap_height - h) // 2
            overlap_row[y0:y0+h, x0:x1] = overlap_img
            cv2.rectangle(overlap_row, (x0, y0), (x1-1, y0+h-1), (255,255,255), 4)
    # 拼接三行
    gap1 = 20
    gap2 = 40
    total_height = label_height + canvas_height + gap1 + max_overlay_height + gap2 + max_overlap_height
    final_img = np.ones((total_height, canvas_width, 3), dtype=np.uint8) * 40
    y = 0
    final_img[y:y+label_height, :, :] = label_img
    y += label_height
    main_y0 = label_height
    final_img[y:y+canvas_height, :, :] = canvas
    y += canvas_height
    y += gap1
    overlay_y0 = y  # <-- 这里定义overlay_y0
    final_img[y:y+max_overlay_height, :, :] = overlay_row
    y += max_overlay_height
    y += gap2
    final_img[y:y+max_overlap_height, :, :] = overlap_row

    # ====== PIL画连线 ======
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(final_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    # 计算第一排每个图片的下边中点
    first_row_bottoms = []
    for i in range(n_sections):
        x0 = offsets[i]
        w = widths[i]
        cx = x0 + w//2
        cy = main_y0 + heights[i]  # 下边y
        first_row_bottoms.append((cx, cy))
    # 计算第二排每个overlay块的上边中点
    overlay_top_centers = []
    for i in range(n_sections-1):
        x0 = overlay_x0_list[i]
        w = overlay_w_list[i]
        if x0 is not None and w is not None:
            cx = x0 + w//2
            cy = overlay_y0  # overlay行的最上边y
            overlay_top_centers.append((cx, cy))
        else:
            overlay_top_centers.append(None)
    # 画线：每对相邻图片下边中点连到对应overlay块上边中点
    for i in range(n_sections-1):
        pt1 = first_row_bottoms[i]
        pt2 = first_row_bottoms[i+1]
        pt_overlay = overlay_top_centers[i]
        if pt_overlay is not None:
            draw.line([pt1, pt_overlay], fill=(255,255,255), width=8)
            draw.line([pt2, pt_overlay], fill=(255,255,255), width=8)
    # 计算第二排overlay块的下边中点，第三排overlap块的上边中点
    overlay_bottom_centers = []
    overlap_top_centers = []
    for i in range(n_sections-1):
        x0 = overlay_x0_list[i]
        w = overlay_w_list[i]
        if x0 is not None and w is not None:
            # overlay下边中点
            cx = x0 + w//2
            cy = overlay_y0 + max_overlay_height  # overlay行的最下边y
            overlay_bottom_centers.append((cx, cy))
            # overlap上边中点
            overlap_cy = overlay_y0 + max_overlay_height + gap  # overlap行的最上边y
            overlap_top_centers.append((cx, overlap_cy))
        else:
            overlay_bottom_centers.append(None)
            overlap_top_centers.append(None)
    # 画线：overlay下边中点连到overlap上边中点
    for i in range(n_sections-1):
        pt_overlay_bottom = overlay_bottom_centers[i]
        pt_overlap_top = overlap_top_centers[i]
        if pt_overlay_bottom is not None and pt_overlap_top is not None:
            draw.line([pt_overlay_bottom, pt_overlap_top], fill=(255,255,255), width=8)
    final_img = np.array(pil_img)
    cv2.imwrite(output_path, final_img)
    print(f'Global match lines with overlay and overlap arrows image saved to {output_path}')

def draw_global_match_lines_with_overlap_split_aligned(folder, chain_file, resize=0.15, output_path='global_match_lines_with_overlap_split_aligned.png', gap_px=100, line_width=4):
    # 1. Parse final order from chain.txt
    def parse_chain_file(chain_file_path):
        with open(chain_file_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if 'final single chain' in line and ':' in line:
                chain_part = line.split(':')[1].strip()
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        for line in lines:
            if 'chain' in line and '->' in line and 'section_' in line:
                chain_part = line.split(':')[1].strip() if ':' in line else line
                sections = [s.strip() for s in chain_part.split('->')]
                return sections
        return None

    section_order = parse_chain_file(chain_file)
    if not section_order:
        print('Error: Could not parse section order from chain file')
        return
    print(f'Final order: {section_order}')

    # 2. Load and resize all images
    images = []
    for sec in section_order:
        for ext in ['.png', '.tif']:
            img_path = os.path.join(folder, f'{sec}{ext}')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    images.append(img)
                    break
        else:
            print(f'Image not found for {sec}')
            return
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    n_sections = len(images)
    # offsets
    offsets = [0]
    for w in widths[:-1]:
        offsets.append(offsets[-1] + w + gap_px)
    canvas_width = sum(widths) + gap_px * (len(widths) - 1)
    label_height = 150
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    font_thickness = 8
    # 第一行：section label
    label_img = np.ones((label_height, canvas_width, 3), dtype=np.uint8) * 255
    for i, sec in enumerate(section_order):
        w = widths[i]
        x0 = offsets[i]
        text = sec
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = x0 + (w - text_w) // 2
        text_y = label_height // 2 + text_h // 2
        cv2.putText(label_img, text, (text_x, text_y), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
    # 第一行：主图
    canvas_height = max(heights)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[:h, offsets[i]:offsets[i]+w] = img
    # 计算每对section之间gap的中心坐标
    gap_centers = []
    for i in range(n_sections-1):
        left = offsets[i] + widths[i]
        right = offsets[i+1]
        gap_centers.append((left + right) // 2)
    # 第二行：overlay块
    overlays = []
    overlay_heights = []
    overlap_imgs = []
    overlap_heights = []
    block_widths = []
    block_heights = []
    for i in range(n_sections-1):
        img1 = images[i]
        img2 = images[i+1]
        processed1 = texture_rich_color_invariant_preprocessing(img1)
        processed2 = texture_rich_color_invariant_preprocessing(img2)
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=20)
        kp1, des1 = sift.detectAndCompute(processed1, None)
        kp2, des2 = sift.detectAndCompute(processed2, None)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            block_widths.append(0)
            block_heights.append(0)
            continue
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.85 * n.distance:
                    good_matches.append(m)
        if len(good_matches) < 10:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            block_widths.append(0)
            block_heights.append(0)
            continue
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0, confidence=0.99, maxIters=5000)
        if M is None or inliers is None:
            overlays.append(None)
            overlay_heights.append(0)
            overlap_imgs.append(None)
            overlap_heights.append(0)
            block_widths.append(0)
            block_heights.append(0)
            continue
        # 画绿色match lines在canvas上
        for match, inlier in zip(good_matches, inliers.flatten()):
            if inlier:
                pt1 = np.array(kp1[match.queryIdx].pt) + [offsets[i], 0]
                pt2 = np.array(kp2[match.trainIdx].pt) + [offsets[i+1], 0]
                cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
        # 生成overlay
        h2, w2 = img2.shape[:2]
        aligned_img1 = cv2.warpAffine(img1, M, (w2, h2))
        overlay = np.zeros((h2, w2, 3), dtype=np.uint8)
        overlay[..., 1] = img2[..., 1]
        overlay[..., 2] = aligned_img1[..., 2]
        mask = (aligned_img1.sum(axis=2) > 0)
        overlay[mask, 1] = img2[mask, 1]
        overlay[mask, 2] = aligned_img1[mask, 2]
        overlays.append(overlay)
        overlay_heights.append(h2)
        # 生成overlap alpha blend（只保留overlap区域）
        overlap_mask = (aligned_img1.sum(axis=2) > 0) & (img2.sum(axis=2) > 0)
        alpha_blend = cv2.addWeighted(aligned_img1, 0.5, img2, 0.5, 0)
        overlap_img = np.zeros_like(img2)
        overlap_img[overlap_mask] = alpha_blend[overlap_mask]
        overlap_imgs.append(overlap_img)
        overlap_heights.append(h2)
        block_widths.append(w2)
        block_heights.append(h2)
    # 统一大间隙
    max_overlay_height = max(overlay_heights) if overlay_heights else 0
    max_overlap_height = max(overlap_heights) if overlap_heights else 0
    gap = max(40, int(max_overlay_height / 5))
    # overlay行分块对齐gap中心
    overlay_row = np.ones((max_overlay_height, canvas_width, 3), dtype=np.uint8) * 40
    for i, overlay in enumerate(overlays):
        if overlay is not None:
            h, w = overlay.shape[:2]
            x_center = gap_centers[i]
            x0 = max(0, x_center - w//2)
            x1 = min(canvas_width, x0 + w)
            if x1 - x0 < w:
                overlay = overlay[:, :x1-x0]
            y0 = (max_overlay_height - h) // 2
            overlay_row[y0:y0+h, x0:x1] = overlay
    # overlap行分块对齐到每个section的白色边框内，且和overlay严格对齐
    overlap_row = np.ones((max_overlap_height, canvas_width, 3), dtype=np.uint8) * 40  # 灰色底
    for i, overlap_img in enumerate(overlap_imgs):
        if overlap_img is not None:
            h, w = overlap_img.shape[:2]
            # 目标区域：第i+1个section的白色边框区域，和overlay一致
            sec_x0 = offsets[i+1]
            sec_w = widths[i+1]
            # section区域内先填充为灰色
            overlap_row[:, sec_x0:sec_x0+sec_w] = 40
            # overlap块在section区域内居中
            x0 = sec_x0 + max(0, (sec_w - w)//2)
            x1 = min(sec_x0 + sec_w, x0 + w)
            if x1 - x0 < w:
                overlap_img = overlap_img[:, :x1-x0]
            y0 = (max_overlap_height - h) // 2
            overlap_row[y0:y0+h, x0:x1] = overlap_img
            # section区域画白色边框
            cv2.rectangle(overlap_row, (sec_x0, y0), (sec_x0+sec_w-1, y0+h-1), (255,255,255), 4)
    # 计算overlay块的左上角x0和宽度w，第三排overlap块直接用相同的x0、w
    overlay_row = np.ones((max_overlay_height, canvas_width, 3), dtype=np.uint8) * 40
    overlap_row = np.ones((max_overlap_height, canvas_width, 3), dtype=np.uint8) * 40
    overlay_x0_list = []
    overlay_w_list = []
    for i in range(n_sections-1):
        overlay = overlays[i]
        if overlay is not None:
            h, w = overlay.shape[:2]
            x_center = gap_centers[i]
            x0 = x_center - w//2
            x1 = x0 + w
            # 边界处理
            if x0 < 0:
                overlay = overlay[:, -x0:]
                x0 = 0
            if x1 > canvas_width:
                overlay = overlay[:, :canvas_width-x0]
                x1 = canvas_width
            y0 = (max_overlay_height - h) // 2
            overlay_row[y0:y0+h, x0:x1] = overlay
            cv2.rectangle(overlay_row, (x0, y0), (x1-1, y0+h-1), (255,255,255), 4)
            overlay_x0_list.append(x0)
            overlay_w_list.append(x1-x0)
        else:
            overlay_x0_list.append(None)
            overlay_w_list.append(None)
    # overlap块与overlay块左对齐，宽度一致
    for i in range(n_sections-1):
        overlap_img = overlap_imgs[i]
        x0 = overlay_x0_list[i]
        w = overlay_w_list[i]
        if overlap_img is not None and x0 is not None and w is not None:
            h, ow = overlap_img.shape[:2]
            # 居中贴到overlay块区域，多余裁剪
            if ow > w:
                overlap_img = overlap_img[:, (ow-w)//2:(ow-w)//2+w]
            elif ow < w:
                pad_left = (w-ow)//2
                pad_right = w-ow-pad_left
                overlap_img = cv2.copyMakeBorder(overlap_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=40)
            y0 = (max_overlap_height - h) // 2
            overlap_row[y0:y0+h, x0:x0+w] = overlap_img
            cv2.rectangle(overlap_row, (x0, y0), (x0+w-1, y0+h-1), (255,255,255), 4)
    # 拼接三行
    total_height = label_height + canvas_height + gap + max_overlay_height + gap + max_overlap_height
    final_img = np.ones((total_height, canvas_width, 3), dtype=np.uint8) * 40
    y = 0
    final_img[y:y+label_height, :, :] = label_img
    y += label_height
    main_y0 = label_height
    final_img[y:y+canvas_height, :, :] = canvas
    y += canvas_height
    y += gap
    overlay_y0 = y  # <-- 这里定义overlay_y0
    final_img[y:y+max_overlay_height, :, :] = overlay_row
    y += max_overlay_height
    y += gap
    final_img[y:y+max_overlap_height, :, :] = overlap_row

    # ====== PIL画连线 ======
    from PIL import Image, ImageDraw
    pil_img = Image.fromarray(final_img.astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    # 计算第一排每个图片的下边中点
    first_row_bottoms = []
    for i in range(n_sections):
        x0 = offsets[i]
        w = widths[i]
        cx = x0 + w//2
        cy = main_y0 + heights[i]  # 下边y
        first_row_bottoms.append((cx, cy))
    # 计算第二排每个overlay块的上边中点
    overlay_top_centers = []
    for i in range(n_sections-1):
        x0 = overlay_x0_list[i]
        w = overlay_w_list[i]
        if x0 is not None and w is not None:
            cx = x0 + w//2
            cy = overlay_y0  # overlay行的最上边y
            overlay_top_centers.append((cx, cy))
        else:
            overlay_top_centers.append(None)
    # 画线：每对相邻图片下边中点连到对应overlay块上边中点
    for i in range(n_sections-1):
        pt1 = first_row_bottoms[i]
        pt2 = first_row_bottoms[i+1]
        pt_overlay = overlay_top_centers[i]
        if pt_overlay is not None:
            draw.line([pt1, pt_overlay], fill=(255,255,255), width=line_width)
            draw.line([pt2, pt_overlay], fill=(255,255,255), width=line_width)
    # 计算第二排overlay块的下边中点，第三排overlap块的上边中点
    overlay_bottom_centers = []
    overlap_top_centers = []
    for i in range(n_sections-1):
        x0 = overlay_x0_list[i]
        w = overlay_w_list[i]
        if x0 is not None and w is not None:
            # overlay下边中点
            cx = x0 + w//2
            cy = overlay_y0 + max_overlay_height  # overlay行的最下边y
            overlay_bottom_centers.append((cx, cy))
            # overlap上边中点
            overlap_cy = overlay_y0 + max_overlay_height + gap  # overlap行的最上边y
            overlap_top_centers.append((cx, overlap_cy))
        else:
            overlay_bottom_centers.append(None)
            overlap_top_centers.append(None)
    # 画线：overlay下边中点连到overlap上边中点
    for i in range(n_sections-1):
        pt_overlay_bottom = overlay_bottom_centers[i]
        pt_overlap_top = overlap_top_centers[i]
        if pt_overlay_bottom is not None and pt_overlap_top is not None:
            draw.line([pt_overlay_bottom, pt_overlap_top], fill=(255,255,255), width=line_width)
    final_img = np.array(pil_img)
    cv2.imwrite(output_path, final_img)
    print(f'Global match lines with overlay and overlap (split, aligned) image saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIFT Pairwise Alignment for Microscopy Sections')
    parser.add_argument('--folder', type=str, default='images_png', help='Folder containing section images')
    parser.add_argument('--resize', type=float, default=0.15, help='Downscale factor applied to both images before alignment (0<r<=1, 1=no scaling)')
    parser.add_argument('--chain', type=str, default='images_png/chain_result_1751982901.txt', help='Path to chain.txt for global match lines')
    parser.add_argument('--output', type=str, default='global_match_lines.png', help='Output image path for global match lines')
    parser.add_argument('--mode', type=str, choices=['pairwise', 'global'], default='pairwise', help='Mode: pairwise alignment or global match lines')
    args = parser.parse_args()
    
    if args.mode == 'global':
        draw_global_match_lines(args.folder, args.chain, args.resize, args.output)
    else:
        main() 

       