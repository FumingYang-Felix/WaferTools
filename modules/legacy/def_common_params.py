import os, glob, numpy as np

# === Custom minimal def_common_params for w06_dSection stack ===
print('Loaded custom def_common_params for w06 stack')

# Directory with PNG slices
root_stack = os.path.abspath('squencing/w06_dSection_060_r01_c01')
assert os.path.isdir(root_stack), f'Data directory not found: {root_stack}'

# List PNG filenames (sorted) as region strings (one slice = one region)
_png = sorted([os.path.splitext(f)[0] for f in os.listdir(root_stack) if f.lower().endswith('.png')])
assert _png, 'No PNG files found in data directory'

# -------------------------------------------------------------
# Minimal global constants expected by msemalign scripts
# -------------------------------------------------------------

total_nwafers      = 1
all_wafer_ids      = range(1, total_nwafers+1)
scale_nm           = 4            # nm per pixel, adjust if known
nimages_per_mfov   = 1            # single image per slice
legacy_zen_format  = True         # treat as legacy single-stack mode

# Downsampling / thumbnail placeholders
use_thumbnails_ds  = 0
thumbnail_suffix   = ''
dsthumbnail        = 1

# Region / slice bookkeeping
region_manifest_cnts = [None, len(_png)]
region_include_cnts  = [None, len(_png)]
exclude_regions      = [[], ] * (total_nwafers+1)

# Region strings list-of-lists as expected by msemalign
region_strs_all = [[_png]]

# Stub values for variables imported by scripts (format strings etc.)
keypoints_dill_fn_str       = 'keypoints_wafer{:02d}.dill'
matches_dill_fn_str         = 'matches_wafer{:02d}.dill'
rough_affine_dill_fn_str    = 'rough_affine_wafer{:02d}.dill'
limi_dill_fn_str            = 'limi_wafer{:02d}.dill'
order_txt_fn_str            = os.path.join('rough_alignment', 'wafer{:02d}_region_solved_order.txt')
exclude_txt_fn_str          = os.path.join('rough_alignment', 'wafer{:02d}_region_excludes.txt')

# Placeholders that scripts expect but are not critical for simple stack alignment
lowe_ratio = 0.75
nfeatures = 5000
max_npts_feature_correspondence = 50000
affine_rigid_type = 'rigid'
min_feature_matches = 32
roi_polygon_scales = [None] + [1.] * total_nwafers
matches_iroi_polygon_scales = [None] + [1.] * total_nwafers
rough_residual_threshold_um = 5
min_fit_pts_radial_std_um = 0.5
max_fit_translation_um = 100
rough_bounding_box_xy_spc = [2048,2048,1]
rough_grid_xy_spc = [1024, 1024, 1]
wafer_solver_bbox_xy_spc = [0,0,0,0]
wafer_solver_bbox_trans = [0,0]
custom_roi = [None]*(total_nwafers+1)

tissue_mask_path = ''
tissue_mask_fn_str = ''
tissue_mask_ds = 0
tissue_mask_min_edge_um = 0
tissue_mask_min_hole_edge_um = 0
tissue_mask_bwdist_um = 0

# Thumbnail folders placeholders
thumbnail_subfolders       = ['']
thumbnail_subfolders_order = ['']
debug_plots_subfolder      = 'debug_plots'
region_suffix              = ''

# Downsampling step
dsstep = 0

# -------------------------------------------------------------
# get_paths()  – core hook used across scripts
# -------------------------------------------------------------

def get_paths(wafer_id):
    """Return tuple fields expected by scripts.
    For our simple stack we only need alignment_folder & meta_folder (both = root_stack)."""
    assert wafer_id == 1, 'Only wafer 1 defined in this minimal setup.'
    alignment_folder = root_stack
    meta_folder      = root_stack
    # experiment_folders, thumbnail_folders, protocol_folders are unused in stack mode
    return [], [], [], alignment_folder, meta_folder, []

# Provide dummy functions/vars when scripts try to import
stack_ext = '.png'
use_roi_poly = False
crops_um = None
dsthumbnail = 1

# In case scripts want tissue mask flags
use_tissue_masks = False

affine_rerun = False

keypoints_nworkers_per_process = 1
keypoints_nprocesses           = 1
matches_full                   = False
matches_gpus                   = 0
keypoints_filter_size          = 0   # no preprocessing filter
keypoints_rescale              = 1.0 # keep original scale 

# === Custom minimal def_common_params for w06_dSection stack ===
print('Loaded custom def_common_params for w06 stack')

# Directory with PNG slices
root_stack = os.path.abspath('squencing/w06_dSection_060_r01_c01')
assert os.path.isdir(root_stack), f'Data directory not found: {root_stack}'

# List PNG filenames (sorted) as region strings (one slice = one region)
_png = sorted([os.path.splitext(f)[0] for f in os.listdir(root_stack) if f.lower().endswith('.png')])
assert _png, 'No PNG files found in data directory'

# -------------------------------------------------------------
# Minimal global constants expected by msemalign scripts
# -------------------------------------------------------------

total_nwafers      = 1
all_wafer_ids      = range(1, total_nwafers+1)
scale_nm           = 4            # nm per pixel, adjust if known
nimages_per_mfov   = 1            # single image per slice
legacy_zen_format  = True         # treat as legacy single-stack mode

# Downsampling / thumbnail placeholders
use_thumbnails_ds  = 0
thumbnail_suffix   = ''
dsthumbnail        = 1

# Region / slice bookkeeping
region_manifest_cnts = [None, len(_png)]
region_include_cnts  = [None, len(_png)]
exclude_regions      = [[], ] * (total_nwafers+1)

# Region strings list-of-lists as expected by msemalign
region_strs_all = [[_png]]

# Stub values for variables imported by scripts (format strings etc.)
keypoints_dill_fn_str       = 'keypoints_wafer{:02d}.dill'
matches_dill_fn_str         = 'matches_wafer{:02d}.dill'
rough_affine_dill_fn_str    = 'rough_affine_wafer{:02d}.dill'
limi_dill_fn_str            = 'limi_wafer{:02d}.dill'
order_txt_fn_str            = os.path.join('rough_alignment', 'wafer{:02d}_region_solved_order.txt')
exclude_txt_fn_str          = os.path.join('rough_alignment', 'wafer{:02d}_region_excludes.txt')

# Placeholders that scripts expect but are not critical for simple stack alignment
lowe_ratio = 0.75
nfeatures = 5000
max_npts_feature_correspondence = 50000
affine_rigid_type = 'rigid'
min_feature_matches = 32
roi_polygon_scales = [None] + [1.] * total_nwafers
matches_iroi_polygon_scales = [None] + [1.] * total_nwafers
rough_residual_threshold_um = 5
min_fit_pts_radial_std_um = 0.5
max_fit_translation_um = 100
rough_bounding_box_xy_spc = [2048,2048,1]
rough_grid_xy_spc = [1024, 1024, 1]
wafer_solver_bbox_xy_spc = [0,0,0,0]
wafer_solver_bbox_trans = [0,0]
custom_roi = [None]*(total_nwafers+1)

tissue_mask_path = ''
tissue_mask_fn_str = ''
tissue_mask_ds = 0
tissue_mask_min_edge_um = 0
tissue_mask_min_hole_edge_um = 0
tissue_mask_bwdist_um = 0

# Thumbnail folders placeholders
thumbnail_subfolders       = ['']
thumbnail_subfolders_order = ['']
debug_plots_subfolder      = 'debug_plots'
region_suffix              = ''

# Downsampling step
dsstep = 0

# -------------------------------------------------------------
# get_paths()  – core hook used across scripts
# -------------------------------------------------------------

def get_paths(wafer_id):
    """Return tuple fields expected by scripts.
    For our simple stack we only need alignment_folder & meta_folder (both = root_stack)."""
    assert wafer_id == 1, 'Only wafer 1 defined in this minimal setup.'
    alignment_folder = root_stack
    meta_folder      = root_stack
    # experiment_folders, thumbnail_folders, protocol_folders are unused in stack mode
    return [], [], [], alignment_folder, meta_folder, []

# Provide dummy functions/vars when scripts try to import
stack_ext = '.png'
use_roi_poly = False
crops_um = None
dsthumbnail = 1

# In case scripts want tissue mask flags
use_tissue_masks = False

affine_rerun = False

keypoints_nworkers_per_process = 1
keypoints_nprocesses           = 1
matches_full                   = False
matches_gpus                   = 0
keypoints_filter_size          = 0   # no preprocessing filter
keypoints_rescale              = 1.0 # keep original scale 
 

# === Custom minimal def_common_params for w06_dSection stack ===
print('Loaded custom def_common_params for w06 stack')

# Directory with PNG slices
root_stack = os.path.abspath('squencing/w06_dSection_060_r01_c01')
assert os.path.isdir(root_stack), f'Data directory not found: {root_stack}'

# List PNG filenames (sorted) as region strings (one slice = one region)
_png = sorted([os.path.splitext(f)[0] for f in os.listdir(root_stack) if f.lower().endswith('.png')])
assert _png, 'No PNG files found in data directory'

# -------------------------------------------------------------
# Minimal global constants expected by msemalign scripts
# -------------------------------------------------------------

total_nwafers      = 1
all_wafer_ids      = range(1, total_nwafers+1)
scale_nm           = 4            # nm per pixel, adjust if known
nimages_per_mfov   = 1            # single image per slice
legacy_zen_format  = True         # treat as legacy single-stack mode

# Downsampling / thumbnail placeholders
use_thumbnails_ds  = 0
thumbnail_suffix   = ''
dsthumbnail        = 1

# Region / slice bookkeeping
region_manifest_cnts = [None, len(_png)]
region_include_cnts  = [None, len(_png)]
exclude_regions      = [[], ] * (total_nwafers+1)

# Region strings list-of-lists as expected by msemalign
region_strs_all = [[_png]]

# Stub values for variables imported by scripts (format strings etc.)
keypoints_dill_fn_str       = 'keypoints_wafer{:02d}.dill'
matches_dill_fn_str         = 'matches_wafer{:02d}.dill'
rough_affine_dill_fn_str    = 'rough_affine_wafer{:02d}.dill'
limi_dill_fn_str            = 'limi_wafer{:02d}.dill'
order_txt_fn_str            = os.path.join('rough_alignment', 'wafer{:02d}_region_solved_order.txt')
exclude_txt_fn_str          = os.path.join('rough_alignment', 'wafer{:02d}_region_excludes.txt')

# Placeholders that scripts expect but are not critical for simple stack alignment
lowe_ratio = 0.75
nfeatures = 5000
max_npts_feature_correspondence = 50000
affine_rigid_type = 'rigid'
min_feature_matches = 32
roi_polygon_scales = [None] + [1.] * total_nwafers
matches_iroi_polygon_scales = [None] + [1.] * total_nwafers
rough_residual_threshold_um = 5
min_fit_pts_radial_std_um = 0.5
max_fit_translation_um = 100
rough_bounding_box_xy_spc = [2048,2048,1]
rough_grid_xy_spc = [1024, 1024, 1]
wafer_solver_bbox_xy_spc = [0,0,0,0]
wafer_solver_bbox_trans = [0,0]
custom_roi = [None]*(total_nwafers+1)

tissue_mask_path = ''
tissue_mask_fn_str = ''
tissue_mask_ds = 0
tissue_mask_min_edge_um = 0
tissue_mask_min_hole_edge_um = 0
tissue_mask_bwdist_um = 0

# Thumbnail folders placeholders
thumbnail_subfolders       = ['']
thumbnail_subfolders_order = ['']
debug_plots_subfolder      = 'debug_plots'
region_suffix              = ''

# Downsampling step
dsstep = 0

# -------------------------------------------------------------
# get_paths()  – core hook used across scripts
# -------------------------------------------------------------

def get_paths(wafer_id):
    """Return tuple fields expected by scripts.
    For our simple stack we only need alignment_folder & meta_folder (both = root_stack)."""
    assert wafer_id == 1, 'Only wafer 1 defined in this minimal setup.'
    alignment_folder = root_stack
    meta_folder      = root_stack
    # experiment_folders, thumbnail_folders, protocol_folders are unused in stack mode
    return [], [], [], alignment_folder, meta_folder, []

# Provide dummy functions/vars when scripts try to import
stack_ext = '.png'
use_roi_poly = False
crops_um = None
dsthumbnail = 1

# In case scripts want tissue mask flags
use_tissue_masks = False

affine_rerun = False

keypoints_nworkers_per_process = 1
keypoints_nprocesses           = 1
matches_full                   = False
matches_gpus                   = 0
keypoints_filter_size          = 0   # no preprocessing filter
keypoints_rescale              = 1.0 # keep original scale 

# === Custom minimal def_common_params for w06_dSection stack ===
print('Loaded custom def_common_params for w06 stack')

# Directory with PNG slices
root_stack = os.path.abspath('squencing/w06_dSection_060_r01_c01')
assert os.path.isdir(root_stack), f'Data directory not found: {root_stack}'

# List PNG filenames (sorted) as region strings (one slice = one region)
_png = sorted([os.path.splitext(f)[0] for f in os.listdir(root_stack) if f.lower().endswith('.png')])
assert _png, 'No PNG files found in data directory'

# -------------------------------------------------------------
# Minimal global constants expected by msemalign scripts
# -------------------------------------------------------------

total_nwafers      = 1
all_wafer_ids      = range(1, total_nwafers+1)
scale_nm           = 4            # nm per pixel, adjust if known
nimages_per_mfov   = 1            # single image per slice
legacy_zen_format  = True         # treat as legacy single-stack mode

# Downsampling / thumbnail placeholders
use_thumbnails_ds  = 0
thumbnail_suffix   = ''
dsthumbnail        = 1

# Region / slice bookkeeping
region_manifest_cnts = [None, len(_png)]
region_include_cnts  = [None, len(_png)]
exclude_regions      = [[], ] * (total_nwafers+1)

# Region strings list-of-lists as expected by msemalign
region_strs_all = [[_png]]

# Stub values for variables imported by scripts (format strings etc.)
keypoints_dill_fn_str       = 'keypoints_wafer{:02d}.dill'
matches_dill_fn_str         = 'matches_wafer{:02d}.dill'
rough_affine_dill_fn_str    = 'rough_affine_wafer{:02d}.dill'
limi_dill_fn_str            = 'limi_wafer{:02d}.dill'
order_txt_fn_str            = os.path.join('rough_alignment', 'wafer{:02d}_region_solved_order.txt')
exclude_txt_fn_str          = os.path.join('rough_alignment', 'wafer{:02d}_region_excludes.txt')

# Placeholders that scripts expect but are not critical for simple stack alignment
lowe_ratio = 0.75
nfeatures = 5000
max_npts_feature_correspondence = 50000
affine_rigid_type = 'rigid'
min_feature_matches = 32
roi_polygon_scales = [None] + [1.] * total_nwafers
matches_iroi_polygon_scales = [None] + [1.] * total_nwafers
rough_residual_threshold_um = 5
min_fit_pts_radial_std_um = 0.5
max_fit_translation_um = 100
rough_bounding_box_xy_spc = [2048,2048,1]
rough_grid_xy_spc = [1024, 1024, 1]
wafer_solver_bbox_xy_spc = [0,0,0,0]
wafer_solver_bbox_trans = [0,0]
custom_roi = [None]*(total_nwafers+1)

tissue_mask_path = ''
tissue_mask_fn_str = ''
tissue_mask_ds = 0
tissue_mask_min_edge_um = 0
tissue_mask_min_hole_edge_um = 0
tissue_mask_bwdist_um = 0

# Thumbnail folders placeholders
thumbnail_subfolders       = ['']
thumbnail_subfolders_order = ['']
debug_plots_subfolder      = 'debug_plots'
region_suffix              = ''

# Downsampling step
dsstep = 0

# -------------------------------------------------------------
# get_paths()  – core hook used across scripts
# -------------------------------------------------------------

def get_paths(wafer_id):
    """Return tuple fields expected by scripts.
    For our simple stack we only need alignment_folder & meta_folder (both = root_stack)."""
    assert wafer_id == 1, 'Only wafer 1 defined in this minimal setup.'
    alignment_folder = root_stack
    meta_folder      = root_stack
    # experiment_folders, thumbnail_folders, protocol_folders are unused in stack mode
    return [], [], [], alignment_folder, meta_folder, []

# Provide dummy functions/vars when scripts try to import
stack_ext = '.png'
use_roi_poly = False
crops_um = None
dsthumbnail = 1

# In case scripts want tissue mask flags
use_tissue_masks = False

affine_rerun = False

keypoints_nworkers_per_process = 1
keypoints_nprocesses           = 1
matches_full                   = False
matches_gpus                   = 0
keypoints_filter_size          = 0   # no preprocessing filter
keypoints_rescale              = 1.0 # keep original scale 
 
 
 
 
 
 
 
 
 
 
 
 
 