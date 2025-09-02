#!/usr/bin/env python3
"""Build an aligned image stack from
  • a cleaned pair-wise CSV that stores dx/dy/angle/scale for each pair
  • a chain_result_*.txt that lists the final single chain order
  • a folder of section images (png/tif)

Output: NumPy .npz with the registered stack (N,H,W,C) and the order list.

Usage (example):
    python build_aligned_stack.py \
        --img_dir   w7_png_4k \
        --csv       pairwise_alignment_results-test4_cleaned.csv \
        --chain_txt chain_result_1751982901.txt \
        --out_npz   aligned_stack.npz \
        --out_wh    1000
"""

import math
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2

# ---------------------------------------------------------------------------
# 1. helpers to parse files
# ---------------------------------------------------------------------------

def get_chain_order(txt_path: Path) -> list[str]:
    """Return the section order from STEP 5."""
    with txt_path.open() as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.startswith("STEP 5: FINAL CONNECTION RESULT"):
            chain = lines[i + 2].split(":", 1)[1]
            return [s.strip() for s in chain.split("->")]
    raise ValueError("final chain not found in txt")


def affine_from_params(dx: float, dy: float, angle_deg: float, scale: float) -> np.ndarray:
    """Make 3×3 H from dx/dy/angle/scale, rotating & scaling around image centre later."""
    # note: translation acts AFTER rotate+scale; centre compensation done later
    cos_a, sin_a = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    A = np.array([[ scale * cos_a, -scale * sin_a, dx ],
                  [ scale * sin_a,  scale * cos_a, dy ],
                  [ 0,              0,              1  ]], dtype=float)
    return A


def load_pairwise(csv_path: Path, img_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    """Return {(moving, fixed): H} using dx/dy/angle/scale from CSV."""
    df = pd.read_csv(csv_path)
    cache_wh: dict[str, tuple[int, int]] = {}
    pairs = {}
    for _, r in df.iterrows():
        moving, fixed = r['moving'], r['fixed']  # column names in cleaned CSV

        if moving not in cache_wh:
            img = cv2.imread(str(img_dir / f"{moving}.png"), cv2.IMREAD_UNCHANGED)
            if img is None:
                img = cv2.imread(str(img_dir / f"{moving}.tif"), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(moving)
            cache_wh[moving] = img.shape[1], img.shape[0]
        W, H = cache_wh[moving]
        cx, cy = W / 2, H / 2

        # build H:  T(dx,dy) • C(cx,cy) • R•S • C^-1
        dx, dy, ang, sc = r['dx_px'], r['dy_px'], r['angle_deg'], r['scale']
        cos_a, sin_a = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        RS = np.array([[ sc*cos_a, -sc*sin_a, 0 ],
                       [ sc*sin_a,  sc*cos_a, 0 ],
                       [ 0,         0,        1 ]])
        C  = np.array([[1,0,cx],[0,1,cy],[0,0,1]])
        Cn = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]])
        T  = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
        H  = T @ C @ RS @ Cn

        pairs[(moving, fixed)] = H
        pairs[(fixed, moving)] = np.linalg.inv(H)
    return pairs


# ---------------------------------------------------------------------------
# 2. build global transforms and warp
# ---------------------------------------------------------------------------

def chain_transforms(order: list[str], pair_H: dict) -> list[np.ndarray]:
    Hs = [np.eye(3)]
    for i in range(1, len(order)):
        H_pair = pair_H[(order[i], order[i-1])]
        Hs.append(Hs[-1] @ H_pair)
    return Hs


def warp_stack(order, Hs, img_dir: Path, out_wh=1000):
    stack = []
    for name, H in zip(order, Hs):
        img = cv2.imread(str(img_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imread(str(img_dir / f"{name}.tif"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(name)
        warped = cv2.warpPerspective(img, H, (out_wh, out_wh),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(0,0,0,0))
        stack.append(warped)
    return np.stack(stack)


# ---------------------------------------------------------------------------
# 3. CLI entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Build aligned image stack from dx/dy/angle/scale CSV & chain txt')
    ap.add_argument('--img_dir', required=True, help='folder of images .png/.tif')
    ap.add_argument('--csv', required=True, help='cleaned pairwise CSV containing dx_px,dy_px,angle_deg,scale')
    ap.add_argument('--chain_txt', required=True, help='chain_result_*.txt')
    ap.add_argument('--out_npz', default='aligned_stack.npz')
    ap.add_argument('--out_wh', type=int, default=1000, help='output width/height (square)')
    args = ap.parse_args()

    img_dir  = Path(args.img_dir)
    csv_path = Path(args.csv)
    txt_path = Path(args.chain_txt)

    order = get_chain_order(txt_path)
    print('Chain order:', order)

    pair_H = load_pairwise(csv_path, img_dir)
    Hs     = chain_transforms(order, pair_H)
    stack  = warp_stack(order, Hs, img_dir, args.out_wh)

    np.savez_compressed(args.out_npz, stack=stack, order=order)
    print('Saved', args.out_npz, 'with shape', stack.shape)

    # 保存为多页tif
    try:
        import tifffile
        tif_path = args.out_npz.replace('.npz', '.tif')
        tifffile.imwrite(tif_path, stack)
        print('Saved', tif_path, 'as multi-page tif')
    except ImportError:
        print('tifffile not installed, skipping tif export.')


if __name__ == '__main__':
    main() 