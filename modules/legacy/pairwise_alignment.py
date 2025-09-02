import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import polylabel as shapely_polylabel  # Shapely >=2.0
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# ---------------------------------------------------------------------------
#   Helper functions
# ---------------------------------------------------------------------------

def resample_polygon(poly: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Uniformly resample a (closed) polygon to *n_points* vertices using spline."""
    from scipy.interpolate import splprep, splev  # local import to avoid hard dep if unused

    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def compute_distance_profile(poly: np.ndarray) -> np.ndarray:
    center = poly.mean(axis=0)
    return np.linalg.norm(poly - center, axis=1)

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    return np.concatenate([arr[shift:], arr[:shift]], axis=0)

def align_polygon_by_circular_correlation(poly: np.ndarray, template_profile: np.ndarray) -> np.ndarray:
    """Cyclically shift *poly* vertices to best match *template_profile*."""
    from numpy.fft import fft, ifft
    profile = compute_distance_profile(poly)
    corr = ifft(fft(profile) * np.conj(fft(template_profile))).real
    shift_idx = int(np.argmax(corr))
    return circular_shift(poly, shift_idx)

def procrustes_similarity(src: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (scale, R) so that (src-mean)*scale*R best matches (ref-mean)."""
    src_c = src - src.mean(0)
    ref_c = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(src_c.T @ ref_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace((src_c @ R).T @ ref_c) / np.trace(src_c.T @ src_c)
    return scale, R

def align_to_template(poly: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Similarity-transform *poly* into reference frame of *ref*."""
    c_src = poly.mean(0)
    c_ref = ref.mean(0)
    s, R = procrustes_similarity(poly, ref)
    return (poly - c_src) * s @ R + c_ref

def anchor_point(poly: np.ndarray, tol: float = 1.0) -> np.ndarray:
    if _HAS_SHAPELY:
        pt = shapely_polylabel(SPoly(poly), tolerance=tol)
        return np.array([pt.x, pt.y])
    return poly.mean(axis=0)

# ---------------------------------------------------------------------------
#   Main routine
# ---------------------------------------------------------------------------

def load_polygons(csv_path: Path) -> Tuple[List[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    polys = []
    for pts_str in df['contour_coordinates']:
        pts = np.array(ast.literal_eval(pts_str), dtype=float)
        if pts.ndim == 3:  # [[[]]] wraps
            pts = pts[0]
        polys.append(pts)
    return polys, df

def main():
    parser = argparse.ArgumentParser(description='Pairwise alignment of wafer section masks.')
    parser.add_argument('--csv', required=True, help='Input aligned_polygons.csv')
    parser.add_argument('--out_dir', default='pairwise_figs', help='Directory to save outputs')
    parser.add_argument('--template_idx', type=int, default=0, help='Row index to use as template')
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    polys, df = load_polygons(csv_path)

    # Optional resampling to equal vertex count (uncomment if needed)
    # polys = [resample_polygon(p, n_points=200) for p in polys]

    template_poly = polys[args.template_idx]
    template_profile = compute_distance_profile(template_poly)

    aligned_polys = []
    anchors = []
    for poly in polys:
        poly_shifted = align_polygon_by_circular_correlation(poly, template_profile)
        poly_aligned = align_to_template(poly_shifted, template_poly)
        aligned_polys.append(poly_aligned)
        anchors.append(anchor_point(poly_aligned))

    # -------------------------------------------------------------------
    # Plot overlay figure
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, anch in zip(aligned_polys, anchors):
        ax.add_patch(Polygon(p, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax.plot(*anch, 'r+', ms=4)
    ax.add_patch(Polygon(template_poly, closed=True, fill=False, lw=1.2, color='k'))
    ax.set_aspect('equal')
    ax.set_title('Pairwise alignment overlay')
    fig.savefig(out_dir / 'pairwise_alignment_overlay.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Write updated CSV with new center_coordinates (anchors)
    # -------------------------------------------------------------------
    df_out = df.copy()
    df_out['center_coordinates'] = [list(map(float, pt)) for pt in anchors]
    out_csv = out_dir / (csv_path.stem + '_pairwise_anchor.csv')
    df_out.to_csv(out_csv, index=False)

    print('Overlay figure saved to', out_dir)
    print('Updated CSV written to ', out_csv)


if __name__ == '__main__':
    main() 
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import polylabel as shapely_polylabel  # Shapely >=2.0
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# ---------------------------------------------------------------------------
#   Helper functions
# ---------------------------------------------------------------------------

def resample_polygon(poly: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Uniformly resample a (closed) polygon to *n_points* vertices using spline."""
    from scipy.interpolate import splprep, splev  # local import to avoid hard dep if unused

    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def compute_distance_profile(poly: np.ndarray) -> np.ndarray:
    center = poly.mean(axis=0)
    return np.linalg.norm(poly - center, axis=1)

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    return np.concatenate([arr[shift:], arr[:shift]], axis=0)

def align_polygon_by_circular_correlation(poly: np.ndarray, template_profile: np.ndarray) -> np.ndarray:
    """Cyclically shift *poly* vertices to best match *template_profile*."""
    from numpy.fft import fft, ifft
    profile = compute_distance_profile(poly)
    corr = ifft(fft(profile) * np.conj(fft(template_profile))).real
    shift_idx = int(np.argmax(corr))
    return circular_shift(poly, shift_idx)

def procrustes_similarity(src: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (scale, R) so that (src-mean)*scale*R best matches (ref-mean)."""
    src_c = src - src.mean(0)
    ref_c = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(src_c.T @ ref_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace((src_c @ R).T @ ref_c) / np.trace(src_c.T @ src_c)
    return scale, R

def align_to_template(poly: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Similarity-transform *poly* into reference frame of *ref*."""
    c_src = poly.mean(0)
    c_ref = ref.mean(0)
    s, R = procrustes_similarity(poly, ref)
    return (poly - c_src) * s @ R + c_ref

def anchor_point(poly: np.ndarray, tol: float = 1.0) -> np.ndarray:
    if _HAS_SHAPELY:
        pt = shapely_polylabel(SPoly(poly), tolerance=tol)
        return np.array([pt.x, pt.y])
    return poly.mean(axis=0)

# ---------------------------------------------------------------------------
#   Main routine
# ---------------------------------------------------------------------------

def load_polygons(csv_path: Path) -> Tuple[List[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    polys = []
    for pts_str in df['contour_coordinates']:
        pts = np.array(ast.literal_eval(pts_str), dtype=float)
        if pts.ndim == 3:  # [[[]]] wraps
            pts = pts[0]
        polys.append(pts)
    return polys, df

def main():
    parser = argparse.ArgumentParser(description='Pairwise alignment of wafer section masks.')
    parser.add_argument('--csv', required=True, help='Input aligned_polygons.csv')
    parser.add_argument('--out_dir', default='pairwise_figs', help='Directory to save outputs')
    parser.add_argument('--template_idx', type=int, default=0, help='Row index to use as template')
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    polys, df = load_polygons(csv_path)

    # Optional resampling to equal vertex count (uncomment if needed)
    # polys = [resample_polygon(p, n_points=200) for p in polys]

    template_poly = polys[args.template_idx]
    template_profile = compute_distance_profile(template_poly)

    aligned_polys = []
    anchors = []
    for poly in polys:
        poly_shifted = align_polygon_by_circular_correlation(poly, template_profile)
        poly_aligned = align_to_template(poly_shifted, template_poly)
        aligned_polys.append(poly_aligned)
        anchors.append(anchor_point(poly_aligned))

    # -------------------------------------------------------------------
    # Plot overlay figure
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, anch in zip(aligned_polys, anchors):
        ax.add_patch(Polygon(p, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax.plot(*anch, 'r+', ms=4)
    ax.add_patch(Polygon(template_poly, closed=True, fill=False, lw=1.2, color='k'))
    ax.set_aspect('equal')
    ax.set_title('Pairwise alignment overlay')
    fig.savefig(out_dir / 'pairwise_alignment_overlay.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Write updated CSV with new center_coordinates (anchors)
    # -------------------------------------------------------------------
    df_out = df.copy()
    df_out['center_coordinates'] = [list(map(float, pt)) for pt in anchors]
    out_csv = out_dir / (csv_path.stem + '_pairwise_anchor.csv')
    df_out.to_csv(out_csv, index=False)

    print('Overlay figure saved to', out_dir)
    print('Updated CSV written to ', out_csv)


if __name__ == '__main__':
    main() 
 
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import polylabel as shapely_polylabel  # Shapely >=2.0
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# ---------------------------------------------------------------------------
#   Helper functions
# ---------------------------------------------------------------------------

def resample_polygon(poly: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Uniformly resample a (closed) polygon to *n_points* vertices using spline."""
    from scipy.interpolate import splprep, splev  # local import to avoid hard dep if unused

    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def compute_distance_profile(poly: np.ndarray) -> np.ndarray:
    center = poly.mean(axis=0)
    return np.linalg.norm(poly - center, axis=1)

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    return np.concatenate([arr[shift:], arr[:shift]], axis=0)

def align_polygon_by_circular_correlation(poly: np.ndarray, template_profile: np.ndarray) -> np.ndarray:
    """Cyclically shift *poly* vertices to best match *template_profile*."""
    from numpy.fft import fft, ifft
    profile = compute_distance_profile(poly)
    corr = ifft(fft(profile) * np.conj(fft(template_profile))).real
    shift_idx = int(np.argmax(corr))
    return circular_shift(poly, shift_idx)

def procrustes_similarity(src: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (scale, R) so that (src-mean)*scale*R best matches (ref-mean)."""
    src_c = src - src.mean(0)
    ref_c = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(src_c.T @ ref_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace((src_c @ R).T @ ref_c) / np.trace(src_c.T @ src_c)
    return scale, R

def align_to_template(poly: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Similarity-transform *poly* into reference frame of *ref*."""
    c_src = poly.mean(0)
    c_ref = ref.mean(0)
    s, R = procrustes_similarity(poly, ref)
    return (poly - c_src) * s @ R + c_ref

def anchor_point(poly: np.ndarray, tol: float = 1.0) -> np.ndarray:
    if _HAS_SHAPELY:
        pt = shapely_polylabel(SPoly(poly), tolerance=tol)
        return np.array([pt.x, pt.y])
    return poly.mean(axis=0)

# ---------------------------------------------------------------------------
#   Main routine
# ---------------------------------------------------------------------------

def load_polygons(csv_path: Path) -> Tuple[List[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    polys = []
    for pts_str in df['contour_coordinates']:
        pts = np.array(ast.literal_eval(pts_str), dtype=float)
        if pts.ndim == 3:  # [[[]]] wraps
            pts = pts[0]
        polys.append(pts)
    return polys, df

def main():
    parser = argparse.ArgumentParser(description='Pairwise alignment of wafer section masks.')
    parser.add_argument('--csv', required=True, help='Input aligned_polygons.csv')
    parser.add_argument('--out_dir', default='pairwise_figs', help='Directory to save outputs')
    parser.add_argument('--template_idx', type=int, default=0, help='Row index to use as template')
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    polys, df = load_polygons(csv_path)

    # Optional resampling to equal vertex count (uncomment if needed)
    # polys = [resample_polygon(p, n_points=200) for p in polys]

    template_poly = polys[args.template_idx]
    template_profile = compute_distance_profile(template_poly)

    aligned_polys = []
    anchors = []
    for poly in polys:
        poly_shifted = align_polygon_by_circular_correlation(poly, template_profile)
        poly_aligned = align_to_template(poly_shifted, template_poly)
        aligned_polys.append(poly_aligned)
        anchors.append(anchor_point(poly_aligned))

    # -------------------------------------------------------------------
    # Plot overlay figure
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, anch in zip(aligned_polys, anchors):
        ax.add_patch(Polygon(p, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax.plot(*anch, 'r+', ms=4)
    ax.add_patch(Polygon(template_poly, closed=True, fill=False, lw=1.2, color='k'))
    ax.set_aspect('equal')
    ax.set_title('Pairwise alignment overlay')
    fig.savefig(out_dir / 'pairwise_alignment_overlay.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Write updated CSV with new center_coordinates (anchors)
    # -------------------------------------------------------------------
    df_out = df.copy()
    df_out['center_coordinates'] = [list(map(float, pt)) for pt in anchors]
    out_csv = out_dir / (csv_path.stem + '_pairwise_anchor.csv')
    df_out.to_csv(out_csv, index=False)

    print('Overlay figure saved to', out_dir)
    print('Updated CSV written to ', out_csv)


if __name__ == '__main__':
    main() 
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import argparse
from typing import List, Tuple

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import polylabel as shapely_polylabel  # Shapely >=2.0
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

# ---------------------------------------------------------------------------
#   Helper functions
# ---------------------------------------------------------------------------

def resample_polygon(poly: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Uniformly resample a (closed) polygon to *n_points* vertices using spline."""
    from scipy.interpolate import splprep, splev  # local import to avoid hard dep if unused

    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)

def compute_distance_profile(poly: np.ndarray) -> np.ndarray:
    center = poly.mean(axis=0)
    return np.linalg.norm(poly - center, axis=1)

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    return np.concatenate([arr[shift:], arr[:shift]], axis=0)

def align_polygon_by_circular_correlation(poly: np.ndarray, template_profile: np.ndarray) -> np.ndarray:
    """Cyclically shift *poly* vertices to best match *template_profile*."""
    from numpy.fft import fft, ifft
    profile = compute_distance_profile(poly)
    corr = ifft(fft(profile) * np.conj(fft(template_profile))).real
    shift_idx = int(np.argmax(corr))
    return circular_shift(poly, shift_idx)

def procrustes_similarity(src: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (scale, R) so that (src-mean)*scale*R best matches (ref-mean)."""
    src_c = src - src.mean(0)
    ref_c = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(src_c.T @ ref_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace((src_c @ R).T @ ref_c) / np.trace(src_c.T @ src_c)
    return scale, R

def align_to_template(poly: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Similarity-transform *poly* into reference frame of *ref*."""
    c_src = poly.mean(0)
    c_ref = ref.mean(0)
    s, R = procrustes_similarity(poly, ref)
    return (poly - c_src) * s @ R + c_ref

def anchor_point(poly: np.ndarray, tol: float = 1.0) -> np.ndarray:
    if _HAS_SHAPELY:
        pt = shapely_polylabel(SPoly(poly), tolerance=tol)
        return np.array([pt.x, pt.y])
    return poly.mean(axis=0)

# ---------------------------------------------------------------------------
#   Main routine
# ---------------------------------------------------------------------------

def load_polygons(csv_path: Path) -> Tuple[List[np.ndarray], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    polys = []
    for pts_str in df['contour_coordinates']:
        pts = np.array(ast.literal_eval(pts_str), dtype=float)
        if pts.ndim == 3:  # [[[]]] wraps
            pts = pts[0]
        polys.append(pts)
    return polys, df

def main():
    parser = argparse.ArgumentParser(description='Pairwise alignment of wafer section masks.')
    parser.add_argument('--csv', required=True, help='Input aligned_polygons.csv')
    parser.add_argument('--out_dir', default='pairwise_figs', help='Directory to save outputs')
    parser.add_argument('--template_idx', type=int, default=0, help='Row index to use as template')
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    polys, df = load_polygons(csv_path)

    # Optional resampling to equal vertex count (uncomment if needed)
    # polys = [resample_polygon(p, n_points=200) for p in polys]

    template_poly = polys[args.template_idx]
    template_profile = compute_distance_profile(template_poly)

    aligned_polys = []
    anchors = []
    for poly in polys:
        poly_shifted = align_polygon_by_circular_correlation(poly, template_profile)
        poly_aligned = align_to_template(poly_shifted, template_poly)
        aligned_polys.append(poly_aligned)
        anchors.append(anchor_point(poly_aligned))

    # -------------------------------------------------------------------
    # Plot overlay figure
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    for p, anch in zip(aligned_polys, anchors):
        ax.add_patch(Polygon(p, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax.plot(*anch, 'r+', ms=4)
    ax.add_patch(Polygon(template_poly, closed=True, fill=False, lw=1.2, color='k'))
    ax.set_aspect('equal')
    ax.set_title('Pairwise alignment overlay')
    fig.savefig(out_dir / 'pairwise_alignment_overlay.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------
    # Write updated CSV with new center_coordinates (anchors)
    # -------------------------------------------------------------------
    df_out = df.copy()
    df_out['center_coordinates'] = [list(map(float, pt)) for pt in anchors]
    out_csv = out_dir / (csv_path.stem + '_pairwise_anchor.csv')
    df_out.to_csv(out_csv, index=False)

    print('Overlay figure saved to', out_dir)
    print('Updated CSV written to ', out_csv)


if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 