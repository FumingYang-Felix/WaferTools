#!/usr/bin/env python3
"""em_pairwise_rigid_alignment.py
=================================
Rigid (translation + rotation) pair-wise alignment of EM (electron microscopy) sections.

Given a directory of grayscale images (TIFF/PNG etc.) the script aligns each
image **i** to the previous image **i-1** in the stack using OpenCV's ECC
algorithm (Euclidean model).  It outputs:

1. CSV file with per-image rigid transform parameters (dx, dy, angle_deg).
2. A diagnostic overlay PNG for each pair (blend of fixed & warped moving).
3. Optionally the warped images themselves.

Example usage
-------------
```
python em_pairwise_rigid_alignment.py \
    --img_dir path/to/sections \
    --out_dir align_out \
    --ext tif --save_warped
```
"""
from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Try initialise NVML to check GPU memory
try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_GPU = True
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HAS_GPU = False

# ---------------------------------------------------------------------------
#   Utilities
# ---------------------------------------------------------------------------

def load_images(img_dir: Path, ext: str) -> list[Path]:
    paths = sorted(img_dir.glob(f"*.{ext}"))
    if not paths:
        sys.exit(f"No *.{ext} images found in {img_dir}")
    return paths

# ---------------- Intensity-based (ECC) ----------------

def ecc_rigid_registration(fixed: np.ndarray, moving: np.ndarray, max_iter=3000, eps=1e-7):
    """Return 2×3 Euclidean warp matrix (rotation + translation)."""
    # ECC expects float32, single-channel
    fixed_f = fixed.astype(np.float32)
    moving_f = moving.astype(np.float32)
    # initial warp = identity
    warp = np.eye(2, 3, dtype=np.float32)
    crit, warp = cv2.findTransformECC(
        templateImage=fixed_f,
        inputImage=moving_f,
        warpMatrix=warp,
        motionType=cv2.MOTION_EUCLIDEAN,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps),
        inputMask=None,
        gaussFiltSize=5,
    )
    return warp

def warp_image(img: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR)

def warp_to_params(warp: np.ndarray) -> tuple[float, float, float]:
    """Convert 2×3 Euclidean warp matrix to (dx, dy, angle_deg)."""
    a, b = warp[0, 0], warp[0, 1]
    angle = np.degrees(np.arctan2(b, a))  # CCW
    dx, dy = warp[0, 2], warp[1, 2]
    return dx, dy, angle

# ---------------- Feature-based (ORB / SIFT) -----------------

def orb_rigid_registration(fixed: np.ndarray, moving: np.ndarray, max_features: int = 5000, use_gpu: bool = False):
    """Estimate Euclidean warp using ORB features + RANSAC.

    Returns 2×3 matrix; if not enough matches, returns identity.
    """
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # GPU path
        orb_cuda = cv2.cuda_ORB_create(nfeatures=max_features)
        g1 = cv2.cuda_GpuMat(); g1.upload(fixed)
        g2 = cv2.cuda_GpuMat(); g2.upload(moving)
        kp1, des1 = orb_cuda.detectAndComputeAsync(g1, None)
        kp2, des2 = orb_cuda.detectAndComputeAsync(g2, None)
        kp1 = orb_cuda.convert(kp1)
        kp2 = orb_cuda.convert(kp2)
        des1 = des1.download() if des1 is not None else None
        des2 = des2.download() if des2 is not None else None
        del g1, g2
    else:
        orb = cv2.ORB_create(nfeatures=max_features)
        kp1, des1 = orb.detectAndCompute(fixed, None)
        kp2, des2 = orb.detectAndCompute(moving, None)

    if des1 is None or des2 is None:
        return np.eye(2, 3, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return np.eye(2, 3, dtype=np.float32)

    pts_fixed = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_moving = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts_moving, pts_fixed, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    return M.astype(np.float32)

def sift_rigid_registration(fixed: np.ndarray, moving: np.ndarray, max_features: int = 0):
    """Estimate Euclidean warp using SIFT + FLANN + RANSAC.

    *max_features* =0 keeps default SIFT behaviour (unlimited)
    Requires OpenCV compiled with contrib (xfeatures2d) or >=4.4 built-in SIFT.
    """
    if not hasattr(cv2, "SIFT_create"):
        print("[WARN] OpenCV SIFT unavailable, falling back to ORB.")
        return orb_rigid_registration(fixed, moving)

    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(fixed, None)
    kp2, des2 = sift.detectAndCompute(moving, None)

    if des1 is None or des2 is None:
        return np.eye(2, 3, dtype=np.float32)

    # FLANN matcher for float descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) < 4:
        # insufficient matches, fall back to ORB
        return orb_rigid_registration(fixed, moving)

    pts_fixed = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_moving = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(pts_moving, pts_fixed, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    return M.astype(np.float32)

# ---------------- SSIM helper -----------------

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Return structural similarity between two uint8 images (0-1)."""
    s, _ = ssim(img1, img2, full=True)
    return float(s)

# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rigid pair-wise alignment for EM sections (Euclidean model)")
    parser.add_argument("--img_dir", required=True, help="Directory containing EM section images")
    parser.add_argument("--out_dir", default="em_align_out", help="Directory to save outputs")
    parser.add_argument("--ext", default="tif", help="Image file extension (no dot)")
    parser.add_argument("--resize", type=float, default=0.25, help="Fractional down-sampling factor for registration (0<r<=1)")
    parser.add_argument("--method", choices=["ecc", "orb", "sift"], default="sift", help="Registration method: ecc | orb | sift")
    parser.add_argument("--save_warped", action="store_true", help="Save warped images to out_dir/warped (only chain mode)")
    parser.add_argument("--full_pairwise", action="store_true", help="Compute alignment between ALL image pairs (i<j) instead of chain")
    parser.add_argument("--infer_order", action="store_true", help="After full pairwise, infer linear order via TSP on custom cost")
    parser.add_argument("--lambda_weight", type=float, default=100.0, help="Lambda weight (um) for SSIM term in cost function")
    parser.add_argument("--mu_weight", type=float, default=1.0, help="Mu weight (um) per degree for angle term in cost function")
    parser.add_argument("--gpu", action="store_true", help="Use GPU via OpenCV CUDA when available for full-res overlay generation")
    parser.add_argument("--batch_gpu_pairs", type=int, default=0, help="Number of pairs to keep in GPU memory concurrently when generating best-pair pages (0 = auto from available memory)")
    parser.add_argument("--cpu_workers", type=int, default=1, help="Number of parallel worker processes for full-pairwise alignment (1 = no multiprocessing, 0 = auto cores-1)")
    args = parser.parse_args()

    if args.cpu_workers == 0:
        args.cpu_workers = max(1, os.cpu_count() - 1)

    img_dir = Path(args.img_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"; overlay_dir.mkdir(exist_ok=True)
    if args.save_warped:
        warped_dir = out_dir / "warped"; warped_dir.mkdir(exist_ok=True)

    img_paths = load_images(img_dir, args.ext)
    print(f"Found {len(img_paths)} images … starting alignment")

    if args.full_pairwise:
        # Preload all resized images for speed
        resized_imgs = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (0,0), fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA) for p in img_paths]
        records = []  # (fixed, moving, dx_px, dy_px, d_um, angle, ssim)
        scale = 1.0/args.resize

        # ---------- multiprocessing workers ----------
        def pair_worker(job):
            i_idx, j_idx, fixed_path, moving_path, resize, method, use_gpu = job
            import cv2, numpy as np
            from skimage.metrics import structural_similarity as ssim

            fixed_full = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
            moving_full = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)

            fixed = cv2.resize(fixed_full, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
            moving = cv2.resize(moving_full, (0,0), fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

            if method=="ecc":
                warp=ecc_rigid_registration(fixed,moving)
            elif method=="orb":
                warp=orb_rigid_registration(fixed,moving,use_gpu=use_gpu)
            else:
                warp=sift_rigid_registration(fixed,moving)

            scale_local=1.0/resize
            warp[0,2]*=scale_local; warp[1,2]*=scale_local
            dx,dy,ang = warp_to_params(warp)

            warped_small = cv2.warpAffine(moving, warp, (fixed.shape[1], fixed.shape[0]), flags=cv2.INTER_LINEAR)
            ssim_val = float(ssim(fixed, warped_small, full=False))

            d_um = np.hypot(dx,dy)*0.04
            return (i_idx,j_idx, dx,dy,d_um,ang,ssim_val)

        jobs=[]
        for i in range(len(img_paths)-1):
            for j in range(i+1,len(img_paths)):
                if args.method == "ecc":
                    pass  # placeholder for indentation alignment
                jobs.append((i,j, str(img_paths[i]), str(img_paths[j]), args.resize, args.method, args.gpu and args.method=="orb"))

        if args.cpu_workers>1:
            with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
                for res in as_completed([pool.submit(pair_worker, job) for job in jobs]):
                    i_idx,j_idx,dx,dy,d_um,ang,ssim_val = res.result()
                    records.append((img_paths[i_idx].name, img_paths[j_idx].name, dx,dy,d_um,ang,ssim_val))
                    print(f"{img_paths[j_idx].name} vs {img_paths[i_idx].name}: d={d_um:.2f}µm, ssim={ssim_val:.3f}, ang={ang:.2f}°  ({args.method})")
        else:
            for job in jobs:
                i_idx,j_idx,dx,dy,d_um,ang,ssim_val = pair_worker(job)
                records.append((img_paths[i_idx].name, img_paths[j_idx].name, dx,dy,d_um,ang,ssim_val))
                print(f"{img_paths[j_idx].name} vs {img_paths[i_idx].name}: d={d_um:.2f}µm, ssim={ssim_val:.3f}, ang={ang:.2f}°  ({args.method})")

        df = pd.DataFrame(records, columns=["fixed","moving","dx_px","dy_px","d_um","angle_deg","ssim"])
        csv_path = out_dir/"full_pairwise_rigid_transforms.csv"
        df.to_csv(csv_path, index=False)
        print("Full pairwise alignment complete. CSV saved to", csv_path)

        # ---- Infer order if requested ----
        if args.infer_order:
            print("Inferring linear order via TSP …")
            imgs = sorted({*df.fixed, *df.moving})
            idx = {name:i for i,name in enumerate(imgs)}
            N=len(imgs)
            cost_mat = np.zeros((N,N))
            for _,row in df.iterrows():
                i,j = idx[row.fixed], idx[row.moving]
                cost = row.d_um + args.lambda_weight*(1-row.ssim) + args.mu_weight*abs(row.angle_deg)
                cost_mat[i,j]=cost_mat[j,i]=cost

            G=nx.complete_graph(N)
            for i in range(N):
                for j in range(i+1,N):
                    G[i][j]['weight']=cost_mat[i,j]
            path = nx.approximation.traveling_salesman_problem(G, weight='weight')
            order=[imgs[i] for i in path]
            pd.DataFrame({'filename':order,'z_index':list(range(len(order)))}).to_csv(out_dir/"inferred_order.csv", index=False)
            print("Order written to inferred_order.csv")

            # ---- PDF report ----
            pdf_path = out_dir/"pairwise_report.pdf"
            with PdfPages(pdf_path) as pdf:
                # cost matrix heatmap
                fig, ax = plt.subplots(figsize=(6,5))
                im=ax.imshow(cost_mat, cmap='viridis')
                ax.set_title('Cost matrix (µm)')
                plt.colorbar(im, ax=ax, shrink=0.8)
                pdf.savefig(fig); plt.close(fig)

                # d_um histogram
                fig,ax=plt.subplots()
                ax.hist(df.d_um, bins=50, color='grey'); ax.set_title('Displacement (µm)');
                pdf.savefig(fig); plt.close(fig)

                # SSIM histogram
                fig,ax=plt.subplots()
                ax.hist(df.ssim, bins=50, color='skyblue'); ax.set_title('SSIM distribution');
                pdf.savefig(fig); plt.close(fig)

                # scatter d vs 1-ssim
                fig,ax=plt.subplots()
                ax.scatter(df.d_um, 1-df.ssim, s=10, alpha=0.6)
                ax.set_xlabel('Displacement (µm)'); ax.set_ylabel('1-SSIM');
                ax.set_title('d vs (1-SSIM)')
                pdf.savefig(fig); plt.close(fig)

                # order list page
                fig,ax=plt.subplots(figsize=(6,8))
                ax.axis('off')
                txt="\n".join([f"{k:03d}  {name}" for k,name in enumerate(order)])
                ax.text(0,1,txt, va='top', fontsize=8, family='monospace')
                ax.set_title('Inferred order')
                pdf.savefig(fig); plt.close(fig)

                # -------- Determine GPU batch size ---------
                if args.gpu and _HAS_GPU and cv2.cuda.getCudaEnabledDeviceCount()>0:
                    info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                    free_mb = info.free / (1024*1024)
                    auto_pairs = int(max(1, (free_mb - 1000) // 320))  # 320 MB per pair with 1 GB cushion
                    batch_pairs = args.batch_gpu_pairs if args.batch_gpu_pairs>0 else auto_pairs
                else:
                    batch_pairs = 1

                pair_indices = [(i, int(np.argmin(np.where(np.arange(N)==i, np.inf, cost_mat[i])))) for i in range(N)]

                for chunk_start in range(0, len(pair_indices), batch_pairs):
                    chunk = pair_indices[chunk_start:chunk_start+batch_pairs]
                    gpu_mats = []  # track to release

                    for i,j_best in chunk:
                        # Load images
                        full_i = cv2.imread(str(img_dir / imgs[i]), cv2.IMREAD_GRAYSCALE)
                        full_j = cv2.imread(str(img_dir / imgs[j_best]), cv2.IMREAD_GRAYSCALE)

                        if args.gpu and _HAS_GPU and cv2.cuda.getCudaEnabledDeviceCount()>0:
                            gi = cv2.cuda_GpuMat(); gi.upload(full_i); gpu_mats.append(gi)
                            gj = cv2.cuda_GpuMat(); gj.upload(full_j); gpu_mats.append(gj)
                            overlay_gpu = cv2.cuda.addWeighted(gi, 0.5, gj, 0.5, 0)
                            overlay = overlay_gpu.download()

                            fig_pair, axes = plt.subplots(1,3, figsize=(12,4))
                            axes[0].imshow(full_i, cmap='gray'); axes[0].set_title(imgs[i]); axes[0].axis('off')
                            axes[1].imshow(full_j, cmap='gray'); axes[1].set_title(imgs[j_best]); axes[1].axis('off')
                            axes[2].imshow(overlay, cmap='gray'); axes[2].set_title('Blend 50/50'); axes[2].axis('off')
                        else:
                            fig_pair, axes = plt.subplots(1,2, figsize=(8,4))
                            axes[0].imshow(full_i, cmap='gray'); axes[0].set_title(imgs[i]); axes[0].axis('off')
                            axes[1].imshow(full_j, cmap='gray'); axes[1].set_title(f'Best: {imgs[j_best]}'); axes[1].axis('off')

                        fig_pair.suptitle(f'Best match for {imgs[i]} (cost={cost_mat[i,j_best]:.1f} µm)')
                        pdf.savefig(fig_pair); plt.close(fig_pair)

                    # Release GPU memory after each chunk
                    for gm in gpu_mats:
                        del gm
                    if args.gpu and _HAS_GPU and cv2.cuda.getCudaEnabledDeviceCount()>0:
                        cv2.cuda.resetDevice()

            print('PDF report written to', pdf_path)
    else:
        # Chain mode as before
        transforms = []
        fixed_full = cv2.imread(str(img_paths[0]), cv2.IMREAD_GRAYSCALE)
        fixed = cv2.resize(fixed_full, (0,0), fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        transforms.append((img_paths[0].name, 0.0,0.0,0.0))
        if args.save_warped:
            cv2.imwrite(str((out_dir/"warped")/img_paths[0].name), fixed_full)

        for idx in range(1,len(img_paths)):
            moving_full = cv2.imread(str(img_paths[idx]), cv2.IMREAD_GRAYSCALE)
            moving = cv2.resize(moving_full, (0,0), fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
            if args.method=="ecc":
                warp=ecc_rigid_registration(fixed,moving)
            elif args.method=="orb":
                warp=orb_rigid_registration(fixed,moving,use_gpu=args.gpu)
            else:
                warp=sift_rigid_registration(fixed,moving)
            scale=1.0/args.resize
            warp[0,2]*=scale; warp[1,2]*=scale
            dx,dy,ang=warp_to_params(warp)
            transforms.append((img_paths[idx].name, dx,dy,ang))
            print(f"Aligned {img_paths[idx].name}: dx={dx:.2f}, dy={dy:.2f}, ang={ang:.2f}°  ({args.method})")
            warped_full=warp_image(moving_full, warp)
            blend=cv2.addWeighted(fixed_full,0.5,warped_full,0.5,0)
            cv2.imwrite(str((out_dir/"overlays")/f"overlay_{idx:04d}.png"), blend)
            if args.save_warped:
                cv2.imwrite(str((out_dir/"warped")/img_paths[idx].name), warped_full)
            fixed_full=warped_full.copy()
            fixed=cv2.resize(fixed_full,(0,0),fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        pd.DataFrame(transforms, columns=["filename","dx","dy","angle_deg"]).to_csv(out_dir/"pairwise_rigid_transforms.csv", index=False)
        print("Chain alignment complete. Results in", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR] Unhandled exception:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)         

        sys.exit(1) 