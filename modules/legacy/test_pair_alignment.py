import cv2
import numpy as np
import pandas as pd

# your single pair alignment algorithm
def apply_on_fix(fix_img, moving_img, params):
    dx, dy, angle, scale = params
    h, w = fix_img.shape[:2]
    # 1️⃣ resize
    fix_scaled = cv2.resize(fix_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # 2️⃣ translation
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    fix_trans = cv2.warpAffine(fix_scaled, M_trans, (moving_img.shape[1], moving_img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    # 3️⃣ rotation (around moving image center)
    cx, cy = moving_img.shape[1]/2, moving_img.shape[0]/2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    fix_aligned = cv2.warpAffine(fix_trans, M_rot, (moving_img.shape[1], moving_img.shape[0]), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
    overlay = cv2.addWeighted(moving_img, 0.5, fix_aligned, 0.5, 0)
    return overlay

# read CSV, find the row where fixed is section_10_r01_c01
csv_path = 'images_png/raw_1b13c623adf74f6e88778ac9594e4799_pairwise_alignment_results-test4_cleaned.csv'
df = pd.read_csv(csv_path)
row = df[df['fixed'] == 'section_10_r01_c01'].iloc[0]

fix_name = row['fixed']
moving_name = row['moving']
params = [row['dx_px'], row['dy_px'], row['angle_deg'], row['scale']]

# read images
dir_path = 'images_png'
fix_img = cv2.imread(f'{dir_path}/{fix_name}.png', cv2.IMREAD_UNCHANGED)
if fix_img is None:
    fix_img = cv2.imread(f'{dir_path}/{fix_name}.tif', cv2.IMREAD_UNCHANGED)
moving_img = cv2.imread(f'{dir_path}/{moving_name}.png', cv2.IMREAD_UNCHANGED)
if moving_img is None:
    moving_img = cv2.imread(f'{dir_path}/{moving_name}.tif', cv2.IMREAD_UNCHANGED)

# apply alignment
overlay = apply_on_fix(fix_img, moving_img, params)
cv2.imwrite('pair_overlay.png', overlay)
print('Saved pair_overlay.png') 
