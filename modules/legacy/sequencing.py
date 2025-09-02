import cv2
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import kornia
except ImportError as e:
    raise ImportError(
        "Required package 'kornia' is not installed. "
        "Please install the necessary dependencies by running:\n"
        "    pip install kornia opencv-python torch torchvision tqdm networkx numpy"
    ) from e
from kornia.feature import LoFTR
import tifffile as tiff

def load_images(dir_path):
    pngs = sorted(list(Path(dir_path).rglob("*.png")))
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in pngs]
    return pngs, imgs


def _init_loftr(device="cuda" if torch.cuda.is_available() else "cpu"):
    matcher = LoFTR(pretrained="outdoor").to(device).eval()
    return matcher


def loftr_match(matcher, img1, img2, device):
    t_img1 = torch.from_numpy(img1 / 255.).float()[None, None].to(device)
    t_img2 = torch.from_numpy(img2 / 255.).float()[None, None].to(device)
    with torch.no_grad():
        pred = matcher({"image0": t_img1, "image1": t_img2})
    mk0 = pred["keypoints0"].cpu().numpy()
    mk1 = pred["keypoints1"].cpu().numpy()
    return mk0, mk1


def estimate_translation(pts0, pts1):
    if len(pts0) < 8:
        return (0.0, 0.0), 0.0
    M, inliers = cv2.estimateAffinePartial2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000)
    if M is None:
        return (0.0, 0.0), 0.0
    dx, dy = float(M[0, 2]), float(M[1, 2])
    conf = inliers.mean() if inliers is not None else 0.0
    return (dx, dy), conf


def build_similarity_graph(imgs, matcher, device, conf_th=0.15):
    from itertools import combinations
    g = nx.Graph()
    n = len(imgs)
    g.add_nodes_from(range(n))
    pairs = list(combinations(range(n), 2))
    for i, j in tqdm(pairs, desc="pairwise", total=len(pairs)):
        kpts0, kpts1 = loftr_match(matcher, imgs[i], imgs[j], device)
        (dx, dy), conf = estimate_translation(kpts0, kpts1)
        if conf < conf_th:
            continue
        g.add_edge(i, j, weight=1.0 - conf, shift=(dx, dy))
    if g.number_of_edges() == 0:
        raise RuntimeError("No reliable matches found between any pair of images.")
    return g


def _accumulate_all_shifts(root: int, g: nx.Graph):
    """Accumulate pairwise shifts along *g* starting from *root* (0,0)."""
    shifts = {root: (0.0, 0.0)}
    stack = [root]
    visited = {root}
    while stack:
        cur = stack.pop()
        for nbr in g.neighbors(cur):
            if nbr in visited:
                continue
            dx, dy = g[cur][nbr]["shift"]
            px, py = shifts[cur]
            shifts[nbr] = (px + dx, py + dy)
            visited.add(nbr)
            stack.append(nbr)
    return shifts


def sequence_from_graph(g):
    """Return an ordering of all nodes that best follows the translation axis.
    1. Build MST of *g* to ensure connectivity.
    2. Choose an arbitrary root (node 0) and accumulate shifts for all nodes.
    3. Perform PCA on the 2-D shifts and sort nodes by their projection onto
       the first principal component (approximate slice axis).
    """
    mst = nx.minimum_spanning_tree(g)
    root = 0
    shifts = _accumulate_all_shifts(root, mst)
    # Perform PCA (essentially compute first eigenvector of covariance)
    coords = np.array([shifts[i] for i in sorted(shifts.keys())])
    if len(coords) < 2:
        # Fallback to numeric order
        order = list(range(len(shifts)))
        return order, shifts
    coords_centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, np.argmax(eigvals)]
    projections = coords_centered @ pc1
    order = [node for _, node in sorted(zip(projections, sorted(shifts.keys())))]
    return order, shifts


def save_montage(filename, imgs, order, shifts):
    xs = [shifts[i][0] for i in order]
    ys = [shifts[i][1] for i in order]
    minx, miny = int(min(xs)), int(min(ys))
    xs = [int(x - minx) for x in xs]
    ys = [int(y - miny) for y in ys]
    sizes = [im.shape for im in imgs]
    width = max(x + w for x, (_, w) in zip(xs, sizes))
    height = max(y + h for y, (h, _) in zip(ys, sizes))
    canvas = np.zeros((height, width), dtype=np.uint8)
    for idx, img_idx in enumerate(order):
        x, y = xs[idx], ys[idx]
        h, w = imgs[img_idx].shape
        canvas[y : y + h, x : x + w] = np.maximum(canvas[y : y + h, x : x + w], imgs[img_idx])
    cv2.imwrite(filename, canvas)
    plt.figure(figsize=(12,12))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.title('Montage')
    plt.show()


def save_tiff_stack(filename: str, imgs, order, shifts):
    """Save a multi-page TIFF stack with all slices aligned according to *shifts*.
    Each page has the same dimensions as the full montage canvas so that the
    slices are spatially registered when the stack is viewed/analysed in image
    software (e.g. Fiji, ImageJ). The output is an 8-bit BigTIFF if the canvas
    exceeds the 4 GiB TIFF limit.
    """
    # Compute target canvas size (same as montage)
    xs = [shifts[i][0] for i in order]
    ys = [shifts[i][1] for i in order]
    minx, miny = int(min(xs)), int(min(ys))
    xs = [int(x - minx) for x in xs]
    ys = [int(y - miny) for y in ys]
    sizes = [im.shape for im in imgs]
    width = max(x + w for x, (_, w) in zip(xs, sizes))
    height = max(y + h for y, (h, _) in zip(ys, sizes))
    frames = []
    for idx, img_idx in enumerate(order):
        frame = np.zeros((height, width), dtype=np.uint8)
        x, y = xs[idx], ys[idx]
        h, w = imgs[img_idx].shape
        frame[y : y + h, x : x + w] = imgs[img_idx]
        frames.append(frame)
    print(f"[INFO] Saving {len(frames)} aligned slices to multipage TIFF: {filename}")
    # tifffile will automatically select BigTIFF if the file size requires it
    tiff.imwrite(
        filename,
        np.stack(frames, axis=0),
        compression=None,
    )
    print("[INFO] TIFF stack saved successfully")

# ------------------------------ CLI ENTRY POINT ----------------------------- #

def main():
    """Entry point for command-line usage."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Estimate slice ordering of a stack of PNG images and create a montage."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing *.png images to be processed.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="montage.png",
        help="Output file. Use .png for a single montage image or .tif/.tiff for a multipage aligned stack.",
    )
    parser.add_argument(
        "--conf-threshold",
        "-t",
        type=float,
        default=0.15,
        help="Confidence threshold for accepting pairwise matches (0-1).",
    )
    args = parser.parse_args()

    # Resolve paths
    inp_dir = Path(args.input_dir).expanduser().resolve()
    if not inp_dir.exists() or not inp_dir.is_dir():
        raise FileNotFoundError(f"Input directory '{inp_dir}' does not exist or is not a directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    matcher = _init_loftr(device)

    paths, imgs = load_images(inp_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No PNG images found in directory: {inp_dir}")
    print(f"[INFO] Loaded {len(imgs)} images: {[p.name for p in paths]}")

    graph = build_similarity_graph(imgs, matcher, device, conf_th=args.conf_threshold)
    order, shifts = sequence_from_graph(graph)

    print("[INFO] Estimated slice order (sorted filenames):")
    for idx in order:
        print(f"  {paths[idx].name}")

    out_path = str(args.output)
    if out_path.lower().endswith( (".tif", ".tiff") ):
        save_tiff_stack(out_path, imgs, order, shifts)
    else:
        save_montage(out_path, imgs, order, shifts)
        print(f"[INFO] Montage saved as {args.output}")

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 