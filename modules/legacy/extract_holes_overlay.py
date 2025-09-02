import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.cluster import KMeans


def extract_holes_from_alpha(alpha):
    tissue = alpha > 0
    filled = binary_fill_holes(tissue)
    holes = filled & (~tissue)
    # remove border-touching holes just in case
    label = measure.label(holes, connectivity=2)
    for region in measure.regionprops(label):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == holes.shape[0] or maxc == holes.shape[1]:
            label[label == region.label] = 0
    holes = label > 0
    return holes


def save_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


def overlay_holes(rgb_img, holes_mask, color=(255, 0, 0)):
    over = rgb_img.copy()
    over[holes_mask] = color
    return over


def chroma_map(rgb):
    lab = color.rgb2lab(rgb)
    return np.linalg.norm(lab[..., 1:3], axis=2)


def detect_hole_by_chroma(rgb, tissue_mask, method='kmeans'):
    C = chroma_map(rgb)
    vals = C[tissue_mask]
    if len(vals) == 0:
        return np.zeros_like(tissue_mask, bool)
    if method == 'kmeans':
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        low_label = int(np.argmin(centers))
        labels_full = np.full(C.shape, -1)
        labels_full[tissue_mask] = km.labels_
        hole = labels_full == low_label
    else:  # otsu
        thr = filters.threshold_otsu(vals)
        hole = (C <= thr) & tissue_mask
    # keep largest connected area
    lab = measure.label(hole)
    areas = [(lab == l).sum() for l in range(1, lab.max() + 1)]
    if not areas:
        return np.zeros_like(hole)
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def main():
    parser = argparse.ArgumentParser(description='Extract holes from tissue PNGs and create overlays.')
    parser.add_argument('--tissue_dir', required=True, help='Directory with RGBA tissue images')
    parser.add_argument('--holes_dir', required=True, help='Output dir for hole masks')
    parser.add_argument('--overlay_dir', required=True, help='Output dir for overlays')
    parser.add_argument('--method', choices=['kmeans', 'otsu'], default='kmeans', help='Hole detection method')
    args = parser.parse_args()

    tissue_dir = Path(args.tissue_dir)
    holes_dir = Path(args.holes_dir)
    overlay_dir = Path(args.overlay_dir)
    holes_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    png_paths = sorted(tissue_dir.glob('*.png'))
    for p in png_paths:
        rgba = np.array(Image.open(p))
        if rgba.shape[2] == 4:
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        else:
            # fallback: assume white bg none holes
            rgb = rgba
            alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        tissue_mask = alpha > 0
        holes = detect_hole_by_chroma(rgb, tissue_mask, method=args.method)
        if holes.any():
            save_mask(holes, holes_dir / p.name.replace('.png', '_holes.png'))
            over = overlay_holes(rgb, holes, color=(255, 0, 0))
            Image.fromarray(over).save(overlay_dir / p.name.replace('.png', '_overlay.png'))
        else:
            print(f'No holes found in {p.name}')

    print('Finished extracting holes and overlays')


if __name__ == '__main__':
    main() 
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.cluster import KMeans


def extract_holes_from_alpha(alpha):
    tissue = alpha > 0
    filled = binary_fill_holes(tissue)
    holes = filled & (~tissue)
    # remove border-touching holes just in case
    label = measure.label(holes, connectivity=2)
    for region in measure.regionprops(label):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == holes.shape[0] or maxc == holes.shape[1]:
            label[label == region.label] = 0
    holes = label > 0
    return holes


def save_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


def overlay_holes(rgb_img, holes_mask, color=(255, 0, 0)):
    over = rgb_img.copy()
    over[holes_mask] = color
    return over


def chroma_map(rgb):
    lab = color.rgb2lab(rgb)
    return np.linalg.norm(lab[..., 1:3], axis=2)


def detect_hole_by_chroma(rgb, tissue_mask, method='kmeans'):
    C = chroma_map(rgb)
    vals = C[tissue_mask]
    if len(vals) == 0:
        return np.zeros_like(tissue_mask, bool)
    if method == 'kmeans':
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        low_label = int(np.argmin(centers))
        labels_full = np.full(C.shape, -1)
        labels_full[tissue_mask] = km.labels_
        hole = labels_full == low_label
    else:  # otsu
        thr = filters.threshold_otsu(vals)
        hole = (C <= thr) & tissue_mask
    # keep largest connected area
    lab = measure.label(hole)
    areas = [(lab == l).sum() for l in range(1, lab.max() + 1)]
    if not areas:
        return np.zeros_like(hole)
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def main():
    parser = argparse.ArgumentParser(description='Extract holes from tissue PNGs and create overlays.')
    parser.add_argument('--tissue_dir', required=True, help='Directory with RGBA tissue images')
    parser.add_argument('--holes_dir', required=True, help='Output dir for hole masks')
    parser.add_argument('--overlay_dir', required=True, help='Output dir for overlays')
    parser.add_argument('--method', choices=['kmeans', 'otsu'], default='kmeans', help='Hole detection method')
    args = parser.parse_args()

    tissue_dir = Path(args.tissue_dir)
    holes_dir = Path(args.holes_dir)
    overlay_dir = Path(args.overlay_dir)
    holes_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    png_paths = sorted(tissue_dir.glob('*.png'))
    for p in png_paths:
        rgba = np.array(Image.open(p))
        if rgba.shape[2] == 4:
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        else:
            # fallback: assume white bg none holes
            rgb = rgba
            alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        tissue_mask = alpha > 0
        holes = detect_hole_by_chroma(rgb, tissue_mask, method=args.method)
        if holes.any():
            save_mask(holes, holes_dir / p.name.replace('.png', '_holes.png'))
            over = overlay_holes(rgb, holes, color=(255, 0, 0))
            Image.fromarray(over).save(overlay_dir / p.name.replace('.png', '_overlay.png'))
        else:
            print(f'No holes found in {p.name}')

    print('Finished extracting holes and overlays')


if __name__ == '__main__':
    main() 
 
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.cluster import KMeans


def extract_holes_from_alpha(alpha):
    tissue = alpha > 0
    filled = binary_fill_holes(tissue)
    holes = filled & (~tissue)
    # remove border-touching holes just in case
    label = measure.label(holes, connectivity=2)
    for region in measure.regionprops(label):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == holes.shape[0] or maxc == holes.shape[1]:
            label[label == region.label] = 0
    holes = label > 0
    return holes


def save_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


def overlay_holes(rgb_img, holes_mask, color=(255, 0, 0)):
    over = rgb_img.copy()
    over[holes_mask] = color
    return over


def chroma_map(rgb):
    lab = color.rgb2lab(rgb)
    return np.linalg.norm(lab[..., 1:3], axis=2)


def detect_hole_by_chroma(rgb, tissue_mask, method='kmeans'):
    C = chroma_map(rgb)
    vals = C[tissue_mask]
    if len(vals) == 0:
        return np.zeros_like(tissue_mask, bool)
    if method == 'kmeans':
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        low_label = int(np.argmin(centers))
        labels_full = np.full(C.shape, -1)
        labels_full[tissue_mask] = km.labels_
        hole = labels_full == low_label
    else:  # otsu
        thr = filters.threshold_otsu(vals)
        hole = (C <= thr) & tissue_mask
    # keep largest connected area
    lab = measure.label(hole)
    areas = [(lab == l).sum() for l in range(1, lab.max() + 1)]
    if not areas:
        return np.zeros_like(hole)
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def main():
    parser = argparse.ArgumentParser(description='Extract holes from tissue PNGs and create overlays.')
    parser.add_argument('--tissue_dir', required=True, help='Directory with RGBA tissue images')
    parser.add_argument('--holes_dir', required=True, help='Output dir for hole masks')
    parser.add_argument('--overlay_dir', required=True, help='Output dir for overlays')
    parser.add_argument('--method', choices=['kmeans', 'otsu'], default='kmeans', help='Hole detection method')
    args = parser.parse_args()

    tissue_dir = Path(args.tissue_dir)
    holes_dir = Path(args.holes_dir)
    overlay_dir = Path(args.overlay_dir)
    holes_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    png_paths = sorted(tissue_dir.glob('*.png'))
    for p in png_paths:
        rgba = np.array(Image.open(p))
        if rgba.shape[2] == 4:
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        else:
            # fallback: assume white bg none holes
            rgb = rgba
            alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        tissue_mask = alpha > 0
        holes = detect_hole_by_chroma(rgb, tissue_mask, method=args.method)
        if holes.any():
            save_mask(holes, holes_dir / p.name.replace('.png', '_holes.png'))
            over = overlay_holes(rgb, holes, color=(255, 0, 0))
            Image.fromarray(over).save(overlay_dir / p.name.replace('.png', '_overlay.png'))
        else:
            print(f'No holes found in {p.name}')

    print('Finished extracting holes and overlays')


if __name__ == '__main__':
    main() 
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure
import matplotlib.pyplot as plt
from skimage import color, filters
from sklearn.cluster import KMeans


def extract_holes_from_alpha(alpha):
    tissue = alpha > 0
    filled = binary_fill_holes(tissue)
    holes = filled & (~tissue)
    # remove border-touching holes just in case
    label = measure.label(holes, connectivity=2)
    for region in measure.regionprops(label):
        minr, minc, maxr, maxc = region.bbox
        if minr == 0 or minc == 0 or maxr == holes.shape[0] or maxc == holes.shape[1]:
            label[label == region.label] = 0
    holes = label > 0
    return holes


def save_mask(mask, path):
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img).save(path)


def overlay_holes(rgb_img, holes_mask, color=(255, 0, 0)):
    over = rgb_img.copy()
    over[holes_mask] = color
    return over


def chroma_map(rgb):
    lab = color.rgb2lab(rgb)
    return np.linalg.norm(lab[..., 1:3], axis=2)


def detect_hole_by_chroma(rgb, tissue_mask, method='kmeans'):
    C = chroma_map(rgb)
    vals = C[tissue_mask]
    if len(vals) == 0:
        return np.zeros_like(tissue_mask, bool)
    if method == 'kmeans':
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        low_label = int(np.argmin(centers))
        labels_full = np.full(C.shape, -1)
        labels_full[tissue_mask] = km.labels_
        hole = labels_full == low_label
    else:  # otsu
        thr = filters.threshold_otsu(vals)
        hole = (C <= thr) & tissue_mask
    # keep largest connected area
    lab = measure.label(hole)
    areas = [(lab == l).sum() for l in range(1, lab.max() + 1)]
    if not areas:
        return np.zeros_like(hole)
    keep = 1 + int(np.argmax(areas))
    return lab == keep


def main():
    parser = argparse.ArgumentParser(description='Extract holes from tissue PNGs and create overlays.')
    parser.add_argument('--tissue_dir', required=True, help='Directory with RGBA tissue images')
    parser.add_argument('--holes_dir', required=True, help='Output dir for hole masks')
    parser.add_argument('--overlay_dir', required=True, help='Output dir for overlays')
    parser.add_argument('--method', choices=['kmeans', 'otsu'], default='kmeans', help='Hole detection method')
    args = parser.parse_args()

    tissue_dir = Path(args.tissue_dir)
    holes_dir = Path(args.holes_dir)
    overlay_dir = Path(args.overlay_dir)
    holes_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    png_paths = sorted(tissue_dir.glob('*.png'))
    for p in png_paths:
        rgba = np.array(Image.open(p))
        if rgba.shape[2] == 4:
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        else:
            # fallback: assume white bg none holes
            rgb = rgba
            alpha = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        tissue_mask = alpha > 0
        holes = detect_hole_by_chroma(rgb, tissue_mask, method=args.method)
        if holes.any():
            save_mask(holes, holes_dir / p.name.replace('.png', '_holes.png'))
            over = overlay_holes(rgb, holes, color=(255, 0, 0))
            Image.fromarray(over).save(overlay_dir / p.name.replace('.png', '_overlay.png'))
        else:
            print(f'No holes found in {p.name}')

    print('Finished extracting holes and overlays')


if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 