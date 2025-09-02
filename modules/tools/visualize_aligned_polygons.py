import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import Tuple

try:
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import polylabel as shapely_polylabel  # Shapely >=2.0
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


def load_polygons_csv(csv_path: Path):
    """Return list of (polygon Nx2 array, center (2,), rotation_deg)."""
    df = pd.read_csv(csv_path)
    polys = []
    centers = []
    rotations = []
    for _, row in df.iterrows():
        pts = np.array(ast.literal_eval(row['contour_coordinates']), dtype=float)
        center = np.array(ast.literal_eval(row['center_coordinates']), dtype=float)
        rot = float(row['rotation'])
        polys.append(pts)
        centers.append(center)
        rotations.append(rot)
    return polys, centers, rotations


def plot_original(polys, centers, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for poly, ctr in zip(polys, centers):
        ax.add_patch(Polygon(poly, closed=True, fill=False, lw=0.8, alpha=0.4))
        ax.plot(*ctr, 'r.', ms=4)
    ax.set_aspect('equal')
    ax.set_title('Original sections & centers')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def rotate_points(points: np.ndarray, angle_deg: float, origin: np.ndarray):
    """Rotate points CCW by *angle_deg* around *origin*."""
    theta = np.deg2rad(angle_deg)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (rot_mat @ (points - origin).T).T + origin


def plot_rotated(polys, centers, rotations, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    for poly, ctr, rot in zip(polys, centers, rotations):
        rot_poly = rotate_points(poly, -rot, ctr)  # align to 0°
        ax.add_patch(Polygon(rot_poly, closed=True, fill=False, lw=0.8, alpha=0.4))
        ax.plot(*ctr, 'g.', ms=4)
    ax.set_aspect('equal')
    ax.set_title('Sections rotated to common orientation')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_stacked(polys, centers, rotations, out_path: Path):
    """Overlay *rotated* polygons in their ORIGINAL positions (no translation).
    This lets us visually inspect how far the section centres deviate after border alignment."""
    fig, ax = plt.subplots(figsize=(8, 8))
    # align all rotated polygons by their bounding box top-left so edges overlap
    ref_shift = None
    for poly, ctr, rot in zip(polys, centers, rotations):
        poly_rot = rotate_points(poly, -rot, ctr)
        bb_min = poly_rot.min(axis=0)
        if ref_shift is None:
            # first polygon establishes reference origin
            ref_shift = bb_min
        shift_vec = ref_shift - bb_min
        poly_aligned = poly_rot + shift_vec
        ctr_shifted = ctr + shift_vec
        ax.add_patch(Polygon(poly_aligned, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax.plot(*ctr_shifted, 'r.', ms=3)
    ax.set_aspect('equal')
    ax.set_title('Stacked rotated contours (centres preserved)')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
#   Anchor point (polylabel) utilities                                       
# ---------------------------------------------------------------------------


def anchor_point(poly_np: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Return a robust interior point of a polygon (approximate polylabel).

    • If Shapely with polylabel support is available, use that (max-distance
      from edges).  Otherwise fall back to polygon centroid.
    """
    if _HAS_SHAPELY:
        shp = SPoly(poly_np)
        try:
            # Shapely 2.0 polylabel returns shapely.Point
            pt = shapely_polylabel(shp, tolerance=tol)
        except Exception:
            pt = shp.centroid
        return np.array([pt.x, pt.y])
    else:
        # fallback: simple centroid
        return poly_np.mean(axis=0)


def plot_anchor_stack(polys, centers, rotations, out_path: Path):
    """Overlay rotated contours aligned by bbox and mark *anchor points* (no centres)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ref_poly = None
    anchor_list = []
    sim_params = []  # (c_i, scale, R)
    for poly, ctr, rot in zip(polys, centers, rotations):
        poly_rot = rotate_points(poly, -rot, ctr)

        # establish reference (template) polygon
        if ref_poly is None:
            ref_poly = poly_rot.copy()

        # similarity transform to reference
        scale, Rmat = procrustes_similarity(poly_rot, ref_poly)
        c_i = poly_rot.mean(0)
        c0 = ref_poly.mean(0)
        poly_aligned = (poly_rot - c_i) * scale @ Rmat + c0

        anch = anchor_point(poly_aligned)

        anchor_list.append(anch)
        sim_params.append((c_i, scale, Rmat))

        ax.add_patch(Polygon(poly_aligned, closed=True, fill=False, lw=0.6, alpha=0.3))

    ax.set_aspect('equal')
    ax.set_title('Stacked rotated contours with anchor points (Procrustes)')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # --- additional outputs (cluster & mapping back) ---
    points = np.vstack(anchor_list)
    inliers, centroid = filter_cluster(points)
    plot_cluster(points, inliers, centroid, out_path.parent / 'fig5_anchor_cluster.png')
    anchor_orig_arr = plot_original_with_new_anchor(polys, centers, rotations, sim_params, centroid, out_path.parent / 'fig6_original_with_anchor.png')

    # write updated CSV with new anchor points
    try:
        df = pd.read_csv(csv_path)
        df['center_coordinates'] = [list(pt) for pt in anchor_orig_arr]
        df.to_csv(out_path.parent / 'aligned_polygons_unified_anchor.csv', index=False)
        print('New CSV saved to', out_path.parent / 'aligned_polygons_unified_anchor.csv')
    except Exception as e:
        print('Failed to write new CSV:', e)


# ---------------------------------------------------------------------------
#   Cluster anchor points & map back to original                              
# ---------------------------------------------------------------------------


def filter_cluster(points: np.ndarray, z_th: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (inliers mask, cluster_centroid) using MAD-based filtering."""
    median = np.median(points, axis=0)
    dists = np.linalg.norm(points - median, axis=1)
    mad = np.median(np.abs(dists - np.median(dists))) + 1e-6
    inliers = dists < z_th * mad
    centroid = points[inliers].mean(axis=0)
    return inliers, centroid


def plot_cluster(points, inliers_mask, centroid, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[~inliers_mask, 0], points[~inliers_mask, 1], s=10, color='gray', label='outliers')
    ax.scatter(points[inliers_mask, 0], points[inliers_mask, 1], s=10, color='red', label='inliers')
    ax.scatter(*centroid, s=60, color='blue', marker='x', label='cluster centre')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Anchor point cluster & centroid')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_original_with_new_anchor(polys, centers, rotations, sim_params, centroid_aligned, out_path: Path):
    """Draw original polygons with unified anchor mapped back using stored similarity params."""
    fig, ax = plt.subplots(figsize=(8, 8))
    anchor_orig_list = []
    c0 = centroid_aligned  # same as reference centroid after alignment
    for poly, ctr, rot, param in zip(polys, centers, rotations, sim_params):
        c_i, scale_fac, Rmat = param

        # inverse similarity transform (original method)
        anchor_tmp = (c0 - c0)  # 0 vector placeholder (anchor relative to c0)
        anchor_tmp = anchor_tmp @ Rmat.T / scale_fac + c_i + (centroid_aligned - c0)

        ax.add_patch(Polygon(poly, closed=True, fill=False, lw=0.8, alpha=0.4))
        ax.plot(*anchor_tmp, 'k+', ms=4)
        anchor_orig_list.append(anchor_tmp)

    ax.set_aspect('equal')
    ax.set_title('Original polygons with unified anchor')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return np.array(anchor_orig_list)


def procrustes_similarity(src: np.ndarray, ref: np.ndarray):
    """Return (s,R) so that (src-mean)*s*R best matches (ref-mean)."""
    src_c = src - src.mean(0)
    ref_c = ref - ref.mean(0)
    U, _, Vt = np.linalg.svd(src_c.T @ ref_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    scale = np.trace((src_c @ R).T @ ref_c) / np.trace(src_c.T @ src_c)
    return scale, R


# ---------------------------------------------------------------------------
#   Generalised Procrustes analysis (GPA)                                    
# ---------------------------------------------------------------------------


def generalized_procrustes(polys, max_iter: int = 15, tol: float = 1e-6):
    """Iteratively align **centred** shapes via similarity transforms so that
    they converge to a common mean shape.

    Parameters
    ----------
    polys : list[np.ndarray]
        List of Nx2 arrays (each polygon).  Vertex correspondence is assumed
        (e.g. after resampling & cyclic alignment).
    max_iter : int
        Maximum GPA iterations.
    tol : float
        Convergence tolerance on mean-shape change (Frobenius norm).

    Returns
    -------
    aligned_polys : list[np.ndarray]
        Aligned polygons in the common coordinate frame (centred near origin).
    transforms : list[Tuple[np.ndarray, float, np.ndarray]]
        Per-polygon (translation, scale, rotation) so that:
            poly_aligned = (poly - trans) * scale @ R
    """
    # Start with first polygon as reference mean shape — centred & unit-norm
    mean_shape = polys[0] - polys[0].mean(axis=0)
    mean_shape /= np.linalg.norm(mean_shape)

    transforms = [None] * len(polys)
    for _ in range(max_iter):
        aligned_list = []
        # Align each polygon to current mean
        for idx, poly in enumerate(polys):
            c = poly.mean(axis=0)
            scale, R = procrustes_similarity(poly, mean_shape)
            aligned = (poly - c) * scale @ R  # centre -> fit -> centred again
            aligned_list.append(aligned)
            transforms[idx] = (c, scale, R)

        new_mean = np.mean(aligned_list, axis=0)
        new_mean -= new_mean.mean(axis=0)
        new_mean_norm = np.linalg.norm(new_mean)
        if new_mean_norm < 1e-12:
            break
        new_mean /= new_mean_norm

        if np.linalg.norm(new_mean - mean_shape) < tol:
            mean_shape = new_mean
            break
        mean_shape = new_mean

    return aligned_list, transforms


def plot_procrustes_stack(polys, centers, rotations, out_path: Path):
    """Generalised Procrustes alignment → anchor overlay (Fig7) + cluster (Fig8) + per-mask overlay (Fig9)."""

    # Step-1: obtain rotated versions of polygons (orientation unified)
    rotated_polys = [rotate_points(poly, -rot, ctr) for poly, ctr, rot in zip(polys, centers, rotations)]

    # Step-2: Generalised Procrustes Analysis to further fuse masks
    aligned_list, transforms = generalized_procrustes(rotated_polys)

    # Step-3: Figure 7 – overlay aligned masks & individual anchors
    fig, ax = plt.subplots(figsize=(8, 8))
    anchor_pts = []
    for poly_aligned in aligned_list:
        ax.add_patch(Polygon(poly_aligned, closed=True, fill=False, lw=0.6, alpha=0.3))
        anch = anchor_point(poly_aligned)
        anchor_pts.append(anch)
        ax.plot(*anch, 'b+', ms=4)

    ax.set_aspect('equal')
    ax.set_title('Global GPA alignment (Fig7)')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    # Step-4: Figure 8 – anchor cloud (no extra artificial shrinking)
    pts = np.vstack(anchor_pts)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(pts[:, 0], pts[:, 1], s=12, color='blue')
    ax2.set_aspect('equal')
    ax2.set_title('Anchor cluster after GPA (Fig8)')
    fig2.savefig(out_path.parent / 'fig8_anchor_cluster.png', dpi=300)
    plt.close(fig2)

    # Step-5: Figure 9 – overlay masks with anchors (same as Fig7 but separate file)
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    for poly_a, pt in zip(aligned_list, anchor_pts):
        ax3.add_patch(Polygon(poly_a, closed=True, fill=False, lw=0.6, alpha=0.3))
        ax3.plot(*pt, 'b+', ms=4)
    ax3.set_aspect('equal')
    ax3.set_title('Anchors over masks after GPA (Fig9)')
    fig3.savefig(out_path.parent / 'fig9_masks_with_anchors_GPA.png', dpi=300)
    plt.close(fig3)

    # Step-6: translate each polygon so that its anchor coincides with global centroid
    centroid = pts.mean(axis=0)
    shifted_polys = [poly_a + (centroid - pt) for poly_a, pt in zip(aligned_list, anchor_pts)]

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    for poly_s in shifted_polys:
        ax4.add_patch(Polygon(poly_s, closed=True, fill=False, lw=0.6, alpha=0.3))
    ax4.plot(*centroid, 'r*', ms=8, label='Unified Anchor')
    ax4.set_aspect('equal')
    ax4.set_title('Polygons shifted to common anchor (Fig10)')
    ax4.legend()
    fig4.savefig(out_path.parent / 'fig10_shifted_polys_common_anchor.png', dpi=300)
    plt.close(fig4)

    # Step-7: enforce identical contour == mean shape for visual verification
    mean_shape = np.mean(aligned_list, axis=0)

    fig5, ax5 = plt.subplots(figsize=(8, 8))
    for _ in aligned_list:
        ax5.add_patch(Polygon(mean_shape, closed=True, fill=False, lw=0.6, alpha=0.3))
    ax5.set_aspect('equal')
    ax5.set_title('All sections replaced by mean contour (Fig11)')
    fig5.savefig(out_path.parent / 'fig11_mean_shape_all.png', dpi=300)
    plt.close(fig5)

    # -------------------------------------------------------------------
    #   Step-8 : Map *aligned polygons + anchors* back to ORIGINAL coords
    # -------------------------------------------------------------------
    poly_back_list = []
    anchor_back_list = []
    for (aligned_poly, anch, (c_i, scale_i, R_i), poly_orig, ctr, rot) in zip(
            aligned_list, anchor_pts, transforms, polys, centers, rotations):

        # Inverse similarity (back to rotated frame)
        poly_rot_back = (aligned_poly @ R_i.T) / scale_i + c_i
        anch_rot_back = (anch @ R_i.T) / scale_i + c_i

        # Undo initial rotation to return to wafer/global coord
        poly_orig_back = rotate_points(poly_rot_back, rot, ctr)
        anch_orig_back = rotate_points(anch_rot_back[None, :], rot, ctr)[0]

        poly_back_list.append(poly_orig_back)
        anchor_back_list.append(anch_orig_back)

    fig6, ax6 = plt.subplots(figsize=(8, 8))
    for poly_b, anch_b in zip(poly_back_list, anchor_back_list):
        ax6.add_patch(Polygon(poly_b, closed=True, fill=False, lw=0.8, alpha=0.4))
        ax6.plot(*anch_b, 'b+', ms=4)
    ax6.set_aspect('equal')
    ax6.set_title('New masks & anchors in original coordinates (Fig12)')
    fig6.savefig(out_path.parent / 'fig12_new_masks_in_original.png', dpi=300)
    plt.close(fig6)

    # --- Step-9 : forward-mapped GPA-shapes (scaled+rotated) back to wafer ---
    poly_new_list = [rotate_points(poly_a, rot, np.zeros(2)) + ctr
                     for poly_a, ctr, rot in zip(aligned_list, centers, rotations)]
    # All sections now share the same unified anchor (centroid)
    anchor_new_list = [rotate_points(anch[None, :], rot, np.zeros(2))[0] + ctr
                       for anch, ctr, rot in zip(anchor_pts, centers, rotations)]

    fig7, ax7 = plt.subplots(figsize=(8, 8))
    for poly_n, anch_n in zip(poly_new_list, anchor_new_list):
        ax7.add_patch(Polygon(poly_n, closed=True, fill=False, lw=0.8, alpha=0.4))
        ax7.plot(*anch_n, 'r+', ms=4)
    ax7.set_aspect('equal')
    ax7.set_title('GPA-scaled masks & anchors in wafer coords (Fig13)')
    fig7.savefig(out_path.parent / 'fig13_gpa_scaled_masks.png', dpi=300)
    plt.close(fig7)

    # -------------------------------------------------------------------
    #   Step-10 : Longest line (diameter) + its midpoint for each section
    # -------------------------------------------------------------------

    def longest_chord(poly_np: np.ndarray):
        """Return indices (i,j) of vertices with maximum Euclidean distance."""
        dists = np.linalg.norm(poly_np[:, None, :] - poly_np[None, :, :], axis=-1)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        return i, j

    fig8, ax8 = plt.subplots(figsize=(8, 8))

    diameter_midpoints = []
    for poly in polys:
        ax8.add_patch(Polygon(poly, closed=True, fill=False, lw=0.6, alpha=0.3))

        i, j = longest_chord(poly)
        p1, p2 = poly[i], poly[j]
        mid = (p1 + p2) / 2.0
        ax8.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', lw=1.0)
        ax8.plot(*mid, 'b*', ms=6)
        diameter_midpoints.append(mid)

    ax8.set_aspect('equal')
    ax8.set_title('Longest chord & midpoint per section (Fig14)')
    fig8.savefig(out_path.parent / 'fig14_longest_chords.png', dpi=300)
    plt.close(fig8)

    # --- Update CSV with new diameter midpoints (original coord system) ---
    try:
        csv_in = csv_path_global  # set in __main__
        df_orig = pd.read_csv(csv_in)
        if len(df_orig) == len(anchor_back_list):
            df_orig['center_coordinates'] = [list(map(float, pt)) for pt in anchor_back_list]
            out_path_csv = csv_in.with_name(csv_in.stem + '_updated_anchor.csv')
            df_orig.to_csv(out_path_csv, index=False)
            print('Updated anchor CSV written to', out_path_csv)
        else:
            print('Mismatch in row count – CSV not updated.')
    except Exception as e:
        print('Failed to update CSV with new anchor:', e)


def plot_polygons_on_image(polys, centers, img_path: Path, out_path: Path):
    """Overlay polygons & anchor centres on a given base image (Fig15)."""
    if not img_path.exists():
        print(f'Base image not found: {img_path}')
        return

    img = plt.imread(img_path)
    h, w = img.shape[:2]
    # scale figure size so that resolution is reasonable (200 dpi)
    fig_w = max(6, w / 800)
    fig_h = max(6, h / 800)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    for poly in polys:
        ax.add_patch(Polygon(poly, closed=True, fill=False, lw=0.6, edgecolor='yellow', alpha=0.7))
    cx = [c[0] for c in centers]
    cy = [c[1] for c in centers]
    ax.plot(cx, cy, 'r+', ms=4, mew=1.2)
    ax.set_axis_off()
    ax.set_title('Polygons & unified anchors overlay (Fig15)')
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualise aligned polygons.')
    parser.add_argument('--csv', default='aligned_polygons.csv', type=str, help='CSV file path')
    parser.add_argument('--out_dir', default='.', type=str, help='Output directory for figures')
    parser.add_argument('--image', default='w08_250nm_1.0mms_100sections_RGB.png_files.png', type=str,
                        help='Base PNG image onto which polygons will be overlaid')
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    global csv_path_global
    csv_path_global = csv_path
    polys, centers, rotations = load_polygons_csv(csv_path)

    plot_original(polys, centers, out_dir / 'fig1_original_sections.png')
    plot_rotated(polys, centers, rotations, out_dir / 'fig2_rotated_sections.png')
    plot_stacked(polys, centers, rotations, out_dir / 'fig3_stacked_sections.png')
    plot_anchor_stack(polys, centers, rotations, out_dir / 'fig4_anchor_points.png')

    print('Figures saved to', out_dir)

    # Fig7 – global Procrustes overlay
    plot_procrustes_stack(polys, centers, rotations, out_dir / 'fig7_procrustes_overlay.png')

    # Fig15 – overlay polygons & anchors on original image
    img_path = Path(args.image).expanduser().resolve()
    plot_polygons_on_image(polys, centers, img_path, out_dir / 'fig15_overlay_on_image.png')

