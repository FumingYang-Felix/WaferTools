import os
import cv2
import numpy as np
import tifffile

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

def texture_rich_color_invariant_preprocessing(img):
    # 可根据你的主流程替换
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def generate_aligned_tif_stack(folder, chain_file, resize=0.25, output_path='aligned_stack.tif'):
    section_order = parse_chain_file(chain_file)
    if not section_order:
        print('Error: Could not parse section order from chain file')
        return
    print(f'Final order: {section_order}')
    imgs = []
    for sec in section_order:
        for ext in ['.png', '.tif']:
            img_path = os.path.join(folder, f'{sec}{ext}')
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
                    imgs.append(img)
                    break
        else:
            print(f'Image not found for {sec}')
            return
    aligned_imgs = []
    h0, w0 = imgs[0].shape[:2]
    # 第一张累计仿射为单位阵
    M_total = np.eye(3)
    aligned_imgs.append(imgs[0])
    for i in range(1, len(imgs)):
        img1 = imgs[i-1]
        img2 = imgs[i]
        proc1 = texture_rich_color_invariant_preprocessing(img1)
        proc2 = texture_rich_color_invariant_preprocessing(img2)
        sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=20)
        kp1, des1 = sift.detectAndCompute(proc1, None)
        kp2, des2 = sift.detectAndCompute(proc2, None)
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            print(f'Not enough features for alignment at index {i}')
            aligned_imgs.append(img2)
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
            print(f'Not enough good matches for alignment at index {i}')
            aligned_imgs.append(img2)
            continue
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0, confidence=0.99, maxIters=5000)
        if M is None or inliers is None:
            print(f'Alignment failed at index {i}')
            aligned_imgs.append(img2)
            continue
        # 累计仿射到全局
        M3 = np.eye(3)
        M3[:2, :] = M
        M_total = M3 @ M_total
        M_total_2x3 = M_total[:2, :]
        aligned_img2 = cv2.warpAffine(imgs[i], M_total_2x3, (w0, h0))
        aligned_imgs.append(aligned_img2)
    # 保存为tif stack
    tifffile.imwrite(output_path, np.stack(aligned_imgs), photometric='rgb')
    print(f'Aligned tif stack saved to {output_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate locally aligned tif stack')
    parser.add_argument('--folder', type=str, required=True, help='Image folder')
    parser.add_argument('--chain_file', type=str, required=True, help='chain.txt path')
    parser.add_argument('--resize', type=float, default=0.15, help='Resize ratio')
    parser.add_argument('--output', type=str, default='aligned_stack.tif', help='Output tif stack path')
    args = parser.parse_args()
    generate_aligned_tif_stack(args.folder, args.chain_file, args.resize, args.output) 