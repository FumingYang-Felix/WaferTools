#!/usr/bin/env python
"""superglue_order.py

Estimate Z-order of a stack of PNG images by SuperPoint+SuperGlue matching
and solve a TSP on the reciprocal match score distance matrix.

Outputs are written inside a user-given output directory:
    order_prelim.txt      – initial order (one filename per line)
    match_heatmap.pdf     – score heat-map (distance matrix)
    pair_matches.pdf      – one page per image pair with inlier lines drawn

The script can be re-run at any time; it will overwrite previous PDFs.

Example:
    python superglue_order.py \
        --images_dir sequencing/w06_dSection_060_r01_c01 \
        --out_dir   ordering_results \
        --resize 1024 --gpu

You need the official SuperGlue pretrained repo cloned next to this file:
    git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git superglue_repo
and make sure `superglue_repo` is on PYTHONPATH, or install with pip:
    pip install superglue-pretrained-network

"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
import os
from pathlib import Path
import gc
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List, Tuple
from itertools import combinations
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Try importing SuperGlue repo ------------------------------------------------
try:
    from models.matching import Matching  # type: ignore
    from models.utils import frame2tensor  # type: ignore
except ModuleNotFoundError:
    raise SystemExit("Cannot import models.matching. Please clone the SuperGluePretrainedNetwork repo and add it to PYTHONPATH, or pip-install it (see script header).")

# ----------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------

def enhance_contrast_clahe(img: np.ndarray, clipLimit: float = 3.0, tileGridSize: Tuple[int, int] = (12, 12)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img.astype(np.uint8))


def load_and_preprocess_pngs(directory: Path, resize_thr: int = 2048) -> Tuple[List[np.ndarray], List[Path]]:
    paths = sorted(directory.glob('*.png'))
    valid_imgs: List[np.ndarray] = []
    valid_paths: List[Path] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'[warn] cannot read {p.name}')
            continue
        m = img.mean()
        if m < 20 or m > 250:
            print(f'[skip] blank-ish {p.name} (mean={m:.1f})')
            continue
        h, w = img.shape
        mx = max(h, w)
        if mx > resize_thr:
            scale = resize_thr / mx
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img = enhance_contrast_clahe(img)
        valid_imgs.append(img)
        valid_paths.append(p)
    if not valid_imgs:
        raise RuntimeError(f'No usable PNGs in {directory}')
    print(f'[info] loaded {len(valid_imgs)} images')
    return valid_imgs, valid_paths


def build_superglue_matcher(model: str = 'outdoor', max_kp: int = 4096):
    """Return (matcher, max_kp) pair."""
    cfg = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.0005,
            'max_keypoints': max_kp,
        },
        'superglue': {
            'weights': model,
        }
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matcher = Matching(cfg).eval().to(device)
    return matcher, max_kp


def match_pair(matcher: Matching, img0: np.ndarray, img1: np.ndarray, device: str = 'cuda') -> Tuple[int, np.ndarray, np.ndarray]:
    """Return (num_matches, kpt0[N,2], kpt1[N,2])"""
    data = {
        'image0': frame2tensor(img0, device),
        'image1': frame2tensor(img1, device)
    }
    with torch.no_grad():
        pred = matcher(data)
    m = pred['matches0'][0].cpu().numpy()
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    valid = m > -1
    idx0 = np.where(valid)[0]
    idx1 = m[valid]
    num = idx0.size
    return num, kpts0[idx0], kpts1[idx1]


def compute_distance_matrix(imgs: List[np.ndarray], matcher: Matching, max_kp: int, pdf_pairs: PdfPages | None = None) -> np.ndarray:
    N = len(imgs)
    D = np.zeros((N, N), dtype=np.float32)
    device = next(matcher.parameters()).device.type
    total_pairs = N * (N - 1) // 2
    pair_iter = tqdm(combinations(range(N), 2), total=total_pairs, desc='Pair matching')
    for i, j in pair_iter:
        num, kp0, kp1 = match_pair(matcher, imgs[i], imgs[j], device)
        score = num / max_kp if max_kp > 0 else 0
        D[i, j] = D[j, i] = 1.0 - score
        pair_iter.set_postfix(pair=f'{i}-{j}', matches=num)
        # visualise ------------------------------------------------------
        if pdf_pairs is not None:
            viz = draw_matches_side_by_side(imgs[i], imgs[j], kp0, kp1, num=max(100, num))
            fig = plt.figure(figsize=(8, 4))
            plt.imshow(viz, cmap='gray')
            plt.axis('off')
            plt.title(f'{i} ↔ {j}: {num} matches')
            pdf_pairs.savefig(fig)
            plt.close(fig)
        # housekeeping ---------------------------------------------------
        torch.cuda.empty_cache()
        gc.collect()
    return D


def draw_matches_side_by_side(img0: np.ndarray, img1: np.ndarray, kp0: np.ndarray, kp1: np.ndarray, num: int = 100) -> np.ndarray:
    """Return a concatenated image with random subset of matches drawn."""
    # scale both to same height
    h = max(img0.shape[0], img1.shape[0])
    s0 = h / img0.shape[0]
    s1 = h / img1.shape[0]
    img0_r = cv2.resize(img0, (int(img0.shape[1] * s0), h))
    img1_r = cv2.resize(img1, (int(img1.shape[1] * s1), h))
    kp0_s = kp0 * s0
    kp1_s = kp1 * s1
    gap = 10
    canvas = np.full((h, img0_r.shape[1] + img1_r.shape[1] + gap), 255, dtype=np.uint8)
    canvas[:, :img0_r.shape[1]] = img0_r
    canvas[:, img0_r.shape[1] + gap:] = img1_r
    # pick subset
    if kp0_s.shape[0] > num:
        sel = np.random.choice(kp0_s.shape[0], num, replace=False)
        kp0_s = kp0_s[sel]
        kp1_s = kp1_s[sel]
    # draw lines
    for p0, p1 in zip(kp0_s, kp1_s):
        pt0 = tuple(np.round(p0).astype(int))
        pt1 = tuple(np.round(p1 + np.array([img0_r.shape[1] + gap, 0])).astype(int))
        color = int(np.random.rand() * 200)
        cv2.line(canvas, pt0, pt1, color=int(color), thickness=1)
        cv2.circle(canvas, pt0, 2, int(color), -1)
        cv2.circle(canvas, pt1, 2, int(color), -1)
    return canvas


def solve_tsp(D: np.ndarray) -> List[int]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    N = D.shape[0]
    manager = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_idx: int, to_idx: int) -> int:
        return int(D[manager.IndexToNode(from_idx), manager.IndexToNode(to_idx)] * 10000)

    trans_id = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(trans_id)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError('TSP solver failed')
    order: List[int] = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return order

# ----------------------------------------------------------------------------
# Main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Estimate section order with SuperGlue matching')
    ap.add_argument('--images_dir', required=True, type=str, help='folder with PNG slices')
    ap.add_argument('--out_dir', required=True, type=str, help='output directory for order & pdfs')
    ap.add_argument('--resize', type=int, default=2048, help='max dimension for processing')
    ap.add_argument('--gpu', action='store_true', help='force use GPU (if available)')
    ap.add_argument('--superglue_weights', type=str, default='outdoor')
    args = ap.parse_args()

    img_dir = Path(args.images_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # load images ------------------------------------------------------------
    imgs, paths = load_and_preprocess_pngs(img_dir, resize_thr=args.resize)

    # matcher ----------------------------------------------------------------
    if args.gpu and torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    matcher, max_kp = build_superglue_matcher(model=args.superglue_weights)
    matcher.to(device_str)

    # PDFs -------------------------------------------------------------------
    pdf_match_path = out_dir / 'pair_matches.pdf'
    pdf_heat_path = out_dir / 'match_heatmap.pdf'
    with PdfPages(pdf_match_path) as pdf_pairs:
        D = compute_distance_matrix(imgs, matcher, max_kp=max_kp, pdf_pairs=pdf_pairs)
    print('[info] pair visualisations saved to', pdf_match_path)

    # heatmap PDF
    with PdfPages(pdf_heat_path) as pdf_hm:
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(D, cmap='viridis')
        plt.colorbar(label='distance (1-score)')
        plt.title('SuperGlue distance matrix')
        pdf_hm.savefig(fig)
        plt.close(fig)
    print('[info] heatmap saved to', pdf_heat_path)

    # order -------------------------------------------------------------------
    order = solve_tsp(D)
    order_txt = out_dir / 'order_prelim.txt'
    with order_txt.open('w') as f:
        for idx in order:
            f.write(paths[idx].stem + '\n')
    print('[info] preliminary order →', order_txt)

    # also dump a simple pdf listing order
    txt_pdf = out_dir / 'order_list.pdf'
    with PdfPages(txt_pdf) as pdf:
        fig = plt.figure(figsize=(6, len(order) * 0.2 + 1))
        plt.axis('off')
        lines = [f'{k:02d}. {paths[i].name}' for k, i in enumerate(order)]
        plt.text(0.01, 0.99, '\n'.join(lines), va='top', family='monospace')
        pdf.savefig(fig)
        plt.close(fig)
    print('[info] order list saved to', txt_pdf)

    # convenience: write FEABAS section_order.txt format
    section_order_path = out_dir / 'section_order.txt'
    with section_order_path.open('w') as f:
        for z, i in enumerate(order):
            f.write(f'{z}\t{paths[i].stem}\n')
    print('[info] FEABAS section_order.txt saved to', section_order_path)

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 