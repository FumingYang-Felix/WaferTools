import os
import pickle
import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import base64
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import uuid
from io import BytesIO
from ultralytics import YOLO
from segment_anything import SamPredictor, build_sam, sam_model_registry, SamAutomaticMaskGenerator
import torch
import requests
from section_identification.section_detector import automatic_identification
from sklearn.cluster import KMeans
import warnings
import sys
import csv
from skimage import measure

# ====== Global cache / constants ======
g_cached_masks = None 
RESULTS_DIR = 'Result_masking'          # automatically fall into ./Result_masking/
# allow override via env var WAFER_SAVE_DEBUG_VIS=1, default off
SAVE_DEBUG_VIS = bool(int(os.getenv('WAFER_SAVE_DEBUG_VIS', '0')))
# legacy output switch: write to ./Result_masking when enabled (default OFF now)
ENABLE_LEGACY_RESULTS = bool(int(os.getenv('WAFER_ENABLE_LEGACY_RESULTS', '0')))

SAM_MODEL_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

def check_and_download_sam_weights(model_type):
    path = f'sam_{model_type}.pth'
    if not os.path.exists(path):
        url = SAM_MODEL_URLS[model_type]
        print(f"Downloading {path} from {url} ...")
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {path}")
    return path

# NOTE: keep original interface, call the unified download function internally
def get_sam_checkpoint(model_type):
    return check_and_download_sam_weights(model_type)

def get_auto_params(image_shape):
    h, w = image_shape
    max_dim = max(h, w)
    if max_dim > 3000:
        downsample_ratio = 2000 / max_dim
    else:
        downsample_ratio = 1.0
    patch_size = 2048 if max_dim > 2000 else 1024
    return downsample_ratio, patch_size

def save_and_resize_image(contents, filename, save_dir='uploads', max_dim=2048):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    # Save original image
    original_path = os.path.join(save_dir, 'original_' + filename)
    img.save(original_path)
    # Create thumbnail for display
    thumb_img = img.copy()
    if max(thumb_img.size) > max_dim:
        thumb_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    thumb_path = os.path.join(save_dir, filename)
    thumb_img.save(thumb_path)
    return thumb_path, original_path, thumb_img

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def initialize_sam_model(model_type='vit_l'):
    check_and_download_sam_weights(model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sam_model_registry[model_type](checkpoint=f'sam_{model_type}.pth')
    model.to(device)
    return SamPredictor(model)

def initialize_auto_mask_generator(model_type='vit_l'):
    """keep original function (currently not directly used in main process)"""
    check_and_download_sam_weights(model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sam_model_registry[model_type](checkpoint=f'sam_{model_type}.pth')
    model.to(device)
    return SamAutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500,
    )

def generate_dense_grid_points(image_shape, patch_size):
    """denser grid (keep original name)"""
    height, width = image_shape
    points = []
    grid_h = max(2, height // (patch_size // 2))
    grid_w = max(2, width // (patch_size // 2))
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            y = int(i * height / grid_h)
            x = int(j * width / grid_w)
            points.append([x, y])
    return points

# === add import =========================================
from modules.section_counter.downsampled_sam import DownsampledSAMDetector    # NEW
# ------------------------------------------------------------

# ====== small tool: uniform mask schema ======
def _ensure_seg(m):
    """return uint8 0/1 mask; tolerate 'segmentation'/'mask' two fields"""
    if 'segmentation' in m:
        return m['segmentation'].astype(np.uint8)
    if 'mask' in m:
        return m['mask'].astype(np.uint8)
    raise ValueError("mask dict missing 'segmentation'/'mask'")

def _to_std_mask_dict(seg_bool: np.ndarray, score: float = 0.0):
    """convert binary bool seg → uniform dict (segmentation/area/bbox/score)"""
    seg_u8 = seg_bool.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(seg_u8)
    return {
        'segmentation': seg_bool.astype(bool),
        'area': int(seg_u8.sum()),
        'bbox': [int(x), int(y), int(w), int(h)],
        'score': float(score)
    }


def _border_pixels(seg_u8: np.ndarray, margin: int = 3) -> int:
    h, w = seg_u8.shape[:2]
    m = min(margin, h//2, w//2)
    if m <= 0: return 0
    return int(seg_u8[:m, :].sum() + seg_u8[-m:, :].sum() +
               seg_u8[:, :m].sum() + seg_u8[:, -m:].sum())

def filter_masks_touching_border(masks, margin: int = 3, rel_border_frac: float = 0.002):
    """
    remove mask that "touching the image border": border pixels / area > rel_border_frac
    default threshold is very small (0.2%), can effectively remove wafer edge/edge large blocks, but will not kill normal section.
    """
    filtered = []
    dropped  = 0
    for m in masks:
        seg = _ensure_seg(m).astype(np.uint8)
        area = max(1, int(seg.sum()))
        bp   = _border_pixels(seg, margin)
        if bp == 0 or (bp / area) < rel_border_frac:
            filtered.append(m)
        else:
            dropped += 1
    if dropped:
        print(f"[DEBUG] border-filter dropped {dropped} mask(s)")
    return filtered

def process_image_with_sam(
        image_path: str,
        model_type: str = "vit_l",
        use_auto_generator: bool = True,
        triggered: str | None = None,
        filter_strength: int | None = None
    ):
    """
    unified entry (keep original name/signature):
      • use_auto_generator = True  → use DownsampledSAMDetector (SAM automatic mask generation)
      • use_auto_generator = False → use improved grid point sampling
    """
    # ---------- 1. read image ----------
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"cannot read image: {image_path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width = original_image.shape[:2]

    # calculate thumbnail (only for display, irrelevant to algorithm)
    downsample_ratio, patch_size = get_auto_params((height, width))
    display_image = cv2.resize(
        original_image,
        (int(width * downsample_ratio), int(height * downsample_ratio))
    )

    # ---------- 2. generate masks ----------
    if use_auto_generator:
        print("[INFO] Downsampled SAM automatic segmentation (DownsampledSAMDetector)")
        # adaptive downsample factor: target long edge ~3000 px (reduce adhesion/over-paste)
        long_edge = max(height, width)
        target = 3000
        factor = max(1, int(round(long_edge / target)))

        detector = DownsampledSAMDetector(
            model_key=f"sam1_{model_type}",
            downsample_factor=factor
        )
        masks = detector.detect_masks(
            image_path,
            min_area=800,
            max_area=50000
        )
        # uniform schema
        std_masks = []
        for m in masks:
            seg = _ensure_seg(m).astype(bool)
            std_masks.append(_to_std_mask_dict(seg, score=float(m.get('score', 0.0))))
        masks = std_masks
        print(f"[INFO] Detected {len(masks)} masks (up-sampled back to original resolution)")
    else:
        print("[INFO] Using improved grid point sampling mode")
        predictor = initialize_sam_model(model_type)
        predictor.set_image(original_image)  # FIX: only set once
        grid_points = generate_dense_grid_points(original_image.shape[:2], patch_size)
        pts = np.array(grid_points, dtype=np.float32)
        lbl = np.ones((len(grid_points),), dtype=np.int32)

        # SamPredictor supports multiple points; the returned shape is usually (N, M, H, W)
        masks_np, scores_np, _ = predictor.predict(
            point_coords=pts,
            point_labels=lbl,
            multimask_output=True
        )

        std_masks = []
        # safe expansion (compatible with different return dimensions)
        for i in range(len(grid_points)):
            # take the candidate with the highest score
            sc_i = scores_np[i]
            mi = int(np.argmax(sc_i))
            seg_i = masks_np[i][mi].astype(bool)
            std_masks.append(_to_std_mask_dict(seg_i, score=float(sc_i[mi])))
            if i % 50 == 0:
                print(f"[DEBUG] grid point {i+1}/{len(grid_points)}")
        masks = std_masks

    # ---------- 3. draw thumbnail ----------
    overlay = original_image.copy()

    if len(masks) == 0:
        print("[WARN] no mask detected")
        return {
            "original_image": original_image,
            "display_image": display_image,
            "mask": []
        }

    print("[DEBUG] Painting masks on thumbnail …")
    H, W = height, width
    kept = 0
    for idx, mask in enumerate(masks):
        seg = _ensure_seg(mask)
        m_disp = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)

        overlay[m_disp > 0] = (245, 245, 225)
        cnts, _ = cv2.findContours(m_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (235, 206, 135), 2)

        scale_factor = max(1.0, W / 1024)
        draw_mask_label(overlay, m_disp, idx + 1, scale=scale_factor)

        kept += 1
        if kept % 20 == 0:
            print(f"[DEBUG]   painted {kept} so far")

    # ---------- 4. return ----------
    return {
        "original_image": original_image,
        "display_image": overlay,
        "mask": masks
    }

def generate_grid_points(image_shape, patch_size):
    """keep original function (not directly used)"""
    height, width = image_shape
    points = []
    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            y = int(i * height / grid_h)
            x = int(j * width / grid_w)
            points.append([x, y])
    return points

def filter_mask(mask, eps_value):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > eps_value:
            filtered_contours.append(contour)
    filtered_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filtered_mask, filtered_contours, -1, 1, -1)
    return filtered_mask.astype(bool)

def auto_segment_and_overlay(image_path, model_type='vit_l', apply_filtering=False, eps_value=200, use_auto_generator=False):
    """keep original interface; fix list filtering type problem"""
    try:
        result = process_image_with_sam(image_path, model_type, use_auto_generator)
        if result is None:
            return None
        if apply_filtering and isinstance(result['mask'], list):
            new_masks = []
            for m in result['mask']:
                seg = _ensure_seg(m)
                seg_f = filter_mask(seg, eps_value).astype(bool)
                mm = m.copy()
                mm['segmentation'] = seg_f
                mm['area'] = int(seg_f.sum())
                x, y, w, h = cv2.boundingRect(seg_f.astype(np.uint8))
                mm['bbox'] = [x, y, w, h]
                new_masks.append(mm)
            result['mask'] = new_masks
        return pil_to_base64(Image.fromarray(result['display_image']))
    except Exception as e:
        print(f"Error in auto_segment_and_overlay: {str(e)}")

def process_image_with_auto_generator(image_path, model_type='vit_l'):
    """keep original interface, remove duplicate return"""
    try:
        print("[INFO] start auto mask generator mode...")
        result = process_image_with_sam(image_path, model_type, use_auto_generator=True)
        if result is None:
            return None
        return pil_to_base64(Image.fromarray(result['display_image']))
    except Exception as e:
        print(f"Error in process_image_with_auto_generator: {str(e)}")
        return None

# ====== Layout ======
layout = html.Div([
    html.H2("Section Counter", className="mb-4 text-primary"),
    dbc.Row([
        # Column 1: Upload image
        dbc.Col([
            dcc.Upload(
                id='section-upload-image',
                children=dbc.Button('Upload Image', color='primary', className='w-100 mb-2'),
                multiple=False
            ),
            html.Span(id='section-upload-status', className='status-text')
        ], width=2),

        # Column 2: Expected sections & Models stacked vertically
        dbc.Col([
            html.Div([
                html.Label('Expected sections:', className='me-2'),
                dbc.Input(id='expected-sections', type='number', min=1, step=1, placeholder='e.g. 100 or blank', style={'width': '100%'})
            ], className='mb-2', style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Label('Models:', className='me-2'),
                dcc.Dropdown(
                    id='sam-model-select',
                    options=[
                        {'label': 'Heavy - Best Quality', 'value': 'vit_h'},
                        {'label': 'Light - Balanced', 'value': 'vit_l'},
                        {'label': 'Lighter - Faster', 'value': 'vit_b'}
                    ],
                    value='vit_l',
                    clearable=False,
                    style={'width': '100%'}
                )
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], width=4, className='ms-4'),

        # Column 3: Detection buttons stacked with brackets indicating choice
        dbc.Col([
            html.Div([
                html.Span('「', style={'position': 'absolute', 'left': '-1rem', 'top': '-2px'}, className='fw-bold'),
                dbc.Button('Auto-loading Cache Detection', id='section-auto-detect-btn', color='success', className='w-100')
            ], className='d-flex align-items-center mb-2', style={'position': 'relative'}),
            html.Div([
                dbc.Button('Run New Detection', id='section-force-recompute-btn', color='danger', className='w-100'),
                html.Span('」', style={'position': 'absolute', 'right': '-1rem', 'bottom': '-2px'}, className='fw-bold')
            ], className='d-flex align-items-center', style={'position': 'relative'}),
            html.Div(id='section-progress-status', className='mt-2')
        ], width=3, className='ms-3')
    ], className='mb-2'),
    dbc.Row([
        dbc.Col(dbc.Button('Filtering', id='section-filter-btn', color='info', className='w-100'), width=2),
        dbc.Col([
            html.Div([
                html.Span('Strict', style={'fontSize': '0.95em', 'color': '#888', 'marginRight': '10px', 'minWidth': '48px'}),
                html.Div(
                    dcc.Slider(
                        id='filter-strength',
                        min=0, max=100, step=1, value=50,
                        marks={0:'', 100:''},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    style={'flex': '1', 'minWidth': '180px', 'margin': '0 8px'}
                ),
                html.Span('Lenient', style={'fontSize': '0.95em', 'color': '#888', 'marginLeft': '10px', 'minWidth': '54px'}),
                html.Span(id='filter-strength-label', className='text-center text-muted', style={'fontSize': '0.95em', 'marginLeft': '16px', 'verticalAlign': 'middle'})
            ], style={'display': 'flex', 'alignItems': 'center', 'width': '100%'})
        ], width=6),
        dbc.Col(
            dbc.Button(
                'Export',
                id='section-export-btn',
                color='secondary',
                className='w-100',
                disabled=True
            ),
            width=2
        ),
    ], className='align-items-center mb-1'),

    # FIX: avoid duplicate id, here change the second progress bar id
    html.Div(id='section-progress-status-aux', className='mb-2'),

    html.Div(id='section-export-status', className='text-success mb-2'),
    dcc.Graph(
        id='section-image-graph',
        config={"displaylogo": False, "scrollZoom": True},
        className="graph-container"
    ),
    html.Img(id='section-image-output', style={'display': 'none'}),
    dcc.Store(id='section-image-store'),
    dcc.Store(id='section-mode-store'),
    html.Div(id='custom-legend', className='my-3'),
], className='container py-4')

def register_section_counter_callbacks(app):
    @app.callback(
        [Output('section-upload-status', 'children'),
         Output('section-image-output', 'src'),
         Output('section-image-store', 'data'),
         Output('section-mode-store', 'data'),
         Output('section-export-btn','disabled')],
        [Input('section-upload-image', 'contents'),
         Input('section-auto-detect-btn', 'n_clicks'),
         Input('section-filter-btn', 'n_clicks'),
         Input('section-force-recompute-btn', 'n_clicks')],
        [State('section-upload-image', 'filename'),
         State('section-image-store', 'data'),
         State('filter-strength', 'value'),
         State('sam-model-select', 'value'),
         State('expected-sections', 'value')],
        prevent_initial_call=True
    )
    def upload_auto_filter(contents, auto_n, filter_n, force_n,
                           filename, image_paths,
                           filter_strength, model_type,
                           expected_count):

        # ---------- trigger source ----------
        triggered       = ctx.triggered_id
        force_recompute = (triggered == 'section-force-recompute-btn')

        # ---------- ① upload image ----------
        if triggered == 'section-upload-image':
            if contents and filename:
                thumb_path, original_path, thumb_img = save_and_resize_image(contents, filename)
                return (f"Image uploaded: {filename}",
                        pil_to_base64(thumb_img),
                        (thumb_path, original_path),
                        'original',
                        True)
            return '', None, None, 'original', True

        # ---------- ② must upload first ----------
        if not image_paths:
            return 'Please upload an image first', None, None, 'original', True

        # ---------- ③ parse path ----------
        if isinstance(image_paths, dict):
            thumb_path    = image_paths.get('thumb_path')
            original_path = image_paths.get('original_path')
        elif isinstance(image_paths, (list, tuple)):
            thumb_path, original_path = image_paths
        else:
            thumb_path = original_path = image_paths

        # ---------- ④ three branches ----------
        global g_cached_masks

        # 4-A  Run New Detection / Force Recompute
        if triggered in ['section-auto-detect-btn', 'section-force-recompute-btn']:
            masks = run_section_detection(
                original_path,
                model_type,
                apply_filtering=False,          # no secondary area filtering (uniform in pipeline)
                filter_strength=0,
                expected_count=expected_count,
                force_recompute=force_recompute,
                enable_smoothing=True,
                smoothing_strength=0.005
            )
            g_cached_masks = masks            # update cache

            # save latest masks.pkl & update image_store (legacy location is optional)
            if ENABLE_LEGACY_RESULTS:
                if not os.path.exists(RESULTS_DIR):
                    os.makedirs(RESULTS_DIR)
                masks_path = os.path.join(
                    RESULTS_DIR,
                    f"masks_{uuid.uuid4().hex[:8]}.pkl"
                )
                with open(masks_path, 'wb') as f:
                    pickle.dump(masks, f)
            else:
                masks_path = None

            image_paths = {
                'thumb_path': thumb_path,
                'original_path': original_path,
                'masks_path': masks_path
            }

        # 4-B  Filtering – only use cache
        elif triggered == 'section-filter-btn':
            if g_cached_masks is None:
                return 'Run detection first', dash.no_update, image_paths, dash.no_update, True

            masks = list(g_cached_masks)
            if expected_count:
                masks = select_masks_by_expected_count(masks, expected_count)
                print(f"[DEBUG] Filtering by expected_count={expected_count} -> {len(masks)} kept")
            else:
                areas  = np.array([m['area'] for m in masks])
                median = np.median(areas)
                s      = max(0.0, min(1.0, (filter_strength or 0) / 100))
                lower  = median * (0.2 - 0.15 * s)   # 0 → 20% median 100 → 5%
                upper  = median * (8.0 + 7.0 * s)    # 0 → 8×median 100 → 15×
                masks  = [m for m in masks if lower <= m['area'] <= upper]
                print(f"[DEBUG] Filtering keep {len(masks)}/{len(areas)}")

            # overwrite cache and disk
            if ENABLE_LEGACY_RESULTS:
                if not os.path.exists(RESULTS_DIR):
                    os.makedirs(RESULTS_DIR)
                masks_path = os.path.join(RESULTS_DIR, 'masks.pkl')
                with open(masks_path, "wb") as f:
                    pickle.dump(masks, f)
            else:
                masks_path = None
            g_cached_masks = masks

            image_paths = {
                'thumb_path': thumb_path,
                'original_path': original_path,
                'masks_path': masks_path
            }
        
        # ---------- ⑤ generate thumbnail ----------
        thumb_img = Image.open(thumb_path).convert('RGB')
        base_np   = np.array(thumb_img)
        H, W      = base_np.shape[:2]
        overlay   = base_np.copy()

        # 5-A  if Filtering, do adjustable range (in case slider is changed on UI)
        if triggered == 'section-filter-btn':
            if expected_count:
                masks = select_masks_by_expected_count(masks, expected_count)
            else:
                areas  = np.array([m['area'] for m in masks])
                median = np.median(areas)
                s      = max(0.0, min(1.0, (filter_strength or 0)/100))
                lower  = median * (0.2 - 0.15*s)
                upper  = median * (8.0  + 7.0*s)
                masks  = [m for m in masks if lower <= m['area'] <= upper]
                print(f"[DEBUG] Filtering keep {len(masks)}/{len(areas)}")

        # 5-B  uniform drawing
        for idx, m in enumerate(masks):
            seg = _ensure_seg(m)
            seg_r = cv2.resize(seg.astype(np.uint8), (W, H), cv2.INTER_NEAREST)
            overlay[seg_r > 0] = (245, 245, 225)
            cnts, _ = cv2.findContours(seg_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (235, 206, 135), 2)
            draw_mask_label(overlay, seg_r, idx+1, scale=max(1.0, W/1024))

        # ---------- ⑥ return base64 ----------
        b64 = pil_to_base64(Image.fromarray(overlay.astype(np.uint8)))
        mode = 'filter' if triggered == 'section-filter-btn' else 'auto'
        return "Completed", b64, image_paths, mode, False   

    # Plot overlay
    @app.callback(
        Output('section-image-graph', 'figure'),
        [Input('section-image-output', 'src')],
        prevent_initial_call=True
    )
    def update_graph(img_b64):
        MAX_CANVAS = 900
        if img_b64:
            from PIL import Image
            import base64, io, sys
            print("[DEBUG] update_graph triggered", file=sys.stderr)
            header, encoded = img_b64.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size
            print(f"[DEBUG] Overlay image size: {width}x{height}", file=sys.stderr)
            sys.stderr.flush()

            scale = min(MAX_CANVAS / width, MAX_CANVAS / height, 1.0)
            canvas_w = int(width * scale)
            canvas_h = int(height * scale)

            fig = go.Figure()
            fig.add_layout_image(
                dict(
                    source=img_b64,
                    x=0,
                    y=0,
                    sizex=width,
                    sizey=height,
                    xref="x",
                    yref="y",
                    sizing="stretch",
                    layer="below"
                )
            )
            fig.update_xaxes(
                visible=False,
                range=[0, width],
                scaleanchor="y",
                scaleratio=1
            )
            fig.update_yaxes(
                visible=False,
                range=[height, 0]
            )
            fig.update_layout(
                width=canvas_w,
                height=canvas_h,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return fig

        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False, 'showgrid': False, 'zeroline': False},
            yaxis={'visible': False, 'showgrid': False, 'zeroline': False},
            annotations=[{
                'text': "Please upload an image",
                'xref': "paper", 'yref': "paper",
                'x': 0.5, 'y': 0.5,
                'xanchor': 'center', 'yanchor': 'middle',
                'showarrow': False,
                'font': {'size': 18},
                'align': 'center'
            }],
            height=600,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig

    # Export results
    @app.callback(
        Output('section-export-status', 'children'),
        Input('section-export-btn', 'n_clicks'),
        State('section-image-store', 'data'),
        State('filter-strength', 'value'),
        prevent_initial_call=True
    )
    def export_results(n_clicks, image_store, filter_strength):
        global g_cached_masks

        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        if g_cached_masks is None:
            return "Please run detection first."
        if not image_store or 'original_path' not in image_store:
            return "Original image path missing."

        masks  = list(g_cached_masks)

        # open original image
        original_path = image_store['original_path']
        img   = Image.open(original_path).convert('RGB')
        img_np = np.array(img)
        H, W   = img_np.shape[:2]
        overlay = img_np.copy()

        # write CSV + visualization
        out_rows = []
        for idx, m in enumerate(masks):
            seg = _ensure_seg(m)
            if seg.shape != (H, W):
                seg_r = cv2.resize(seg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                seg_r = seg.astype(np.uint8)

            contours = measure.find_contours(seg_r, 0.5)
            if not contours:
                continue
            contour = max(contours, key=lambda x: x.shape[0])
            if (contour[0] != contour[-1]).any():
                contour = np.vstack([contour, contour[0]])
            contour_list = [[float(c[1]), float(c[0])] for c in contour]

            area = int(seg_r.sum())
            xs, ys = zip(*contour_list)
            centroid_x, centroid_y = float(np.mean(xs)), float(np.mean(ys))
            bbox = [int(min(xs)), int(min(ys)),
                    int(max(xs) - min(xs)), int(max(ys) - min(ys))]

            out_rows.append([
                f'section_{idx+1}',
                str(contour_list),
                area,
                centroid_x,
                centroid_y,
                bbox
            ])

            overlay[seg_r > 0] = (245, 245, 225)
            cnts, _ = cv2.findContours(seg_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (235, 206, 135), 2)
            draw_mask_label(overlay, seg_r, idx+1, scale=max(1.0, W/1024))

        mask_png_path = None
        sections_csv_path = None
        if ENABLE_LEGACY_RESULTS:
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            mask_png_path = os.path.join(RESULTS_DIR, 'mask.png')
            cv2.imwrite(mask_png_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            sections_csv_path = os.path.join(RESULTS_DIR, 'sections.csv')
            with open(sections_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'contour_coordinates',
                                'area', 'centroid_x', 'centroid_y', 'bbox'])
                writer.writerows(out_rows)
        # ---- append writing to unified results directory (does not affect original writing) ----
        try:
            from modules.common.paths import get_run_dir
            from modules.common.io import copy_to, save_meta
            run_dir = get_run_dir('section_counter')
            files_to_copy = []
            if mask_png_path and os.path.exists(mask_png_path):
                files_to_copy.append((mask_png_path, 'mask.png'))
            if sections_csv_path and os.path.exists(sections_csv_path):
                files_to_copy.append((sections_csv_path, 'sections.csv'))
            mp = image_store.get('masks_path') if isinstance(image_store, dict) else None
            if mp and os.path.exists(mp):
                files_to_copy.append((mp, 'masks.pkl'))
            copy_to(run_dir, files_to_copy)
            save_meta(run_dir, {
                'module': 'section_counter',
                'export_count': len(out_rows)
            })
        except Exception:
            pass

        target_desc = RESULTS_DIR if ENABLE_LEGACY_RESULTS else f"results/section_counter/{os.path.basename(run_dir)}"
        return f"✅ Exported {len(out_rows)} sections to {target_desc}"

    @app.callback(
        Output('filter-strength-label', 'children'),
        [Input('filter-strength', 'value')]
    )
    def update_filter_strength_label(value):
        return f'{value}%'

    @app.callback(
        Output('section-progress-status', 'children'),
        [Input('section-auto-detect-btn', 'n_clicks'),
         Input('section-filter-btn', 'n_clicks')],
        [State('section-image-store', 'data')],
        prevent_initial_call=True
    )
    def show_processing_status(auto_n, filter_n, image_path):
        triggered = ctx.triggered_id
        if (triggered == 'section-auto-detect-btn' and auto_n and image_path) or (triggered == 'section-filter-btn' and filter_n and image_path):
            return 'Processing...'
        return ''

    # FIX: write "Completed!" to the second progress bar, avoid duplicate id conflict
    @app.callback(
        Output('section-progress-status-aux', 'children'),
        [Input('section-image-graph', 'figure')],
        prevent_initial_call=True
    )
    def show_completed_status(fig):
        if fig and fig.get('data'):
            return 'Completed!'
        return ''

    # Legend consistent with screen: directly use g_cached_masks
    @app.callback(
        Output('custom-legend', 'children'),
        [Input('section-mode-store', 'data'),
         Input('section-image-store', 'data'),
         Input('filter-strength', 'value')]
    )
    def update_custom_legend(mode, image_paths, filter_strength):
        if mode != 'filter' or not image_paths:
            return ""
        from dash import html
        global g_cached_masks
        masks = g_cached_masks or []
        if not masks:
            return ""

        import plotly.colors
        colors = plotly.colors.qualitative.Plotly

        legend_items = []
        for idx, _ in enumerate(masks):
            color = colors[idx % len(colors)]
            legend_items.append(
                html.Span([
                    html.Span(style={'display': 'inline-block', 'width': '18px', 'height': '8px', 'backgroundColor': color, 'marginRight': '8px'}),
                    f'section {idx+1}'
                ], style={'marginRight': '18px', 'fontSize': '1.1em'})
            )
        legend_div = html.Div(legend_items, style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'})
        return legend_div

def save_masks_visualization(image_path, masks, out_path):
    # Hard-disable debug visualization saving
    return

# ---------------------------------------------------------
# utility function: IoU / NMS-style deduplication
# ---------------------------------------------------------
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else inter / union



def deduplicate_masks(masks, iou_thresh: float = 0.80, keep: str = "auto"):
    """
    keep:
      - 'larger'/'smaller': keep larger/smaller (compatible with old parameters)
      - 'auto': keep the one "closer to the median area of the whole" (recommended, can avoid replacing a complete section with small fragments)
    """
    if not masks:
        return masks
    areas = np.array([m["area"] for m in masks], dtype=float)
    median_area = float(np.median(areas))

    # traverse from large to small, prioritize large area (reduce the probability of being "eaten" by small fragments)
    ordered = sorted(masks, key=lambda m: m["area"], reverse=True)
    kept = []
    for cand in ordered:
        seg_c = _ensure_seg(cand).astype(bool)
        is_dup = False
        for i, k in enumerate(kept):
            seg_k = _ensure_seg(k).astype(bool)
            if _mask_iou(seg_c, seg_k) >= iou_thresh:
                is_dup = True
                if keep == "larger":
                    if cand["area"] > k["area"]:
                        kept[i] = cand
                elif keep == "smaller":
                    if cand["area"] < k["area"]:
                        kept[i] = cand
                else:  # 'auto'
                    # who is closer to the median area keeps who
                    if abs(cand["area"] - median_area) < abs(k["area"] - median_area):
                        kept[i] = cand
                break
        if not is_dup:
            kept.append(cand)
    return kept


from modules.section_counter.downsampled_sam import DownsampledSAMDetector 

def select_masks_by_expected_count(masks, expected_count: int):
    if not masks or not expected_count:
        return masks
    try:
        expected_count = int(expected_count)
    except Exception:
        pass

    if len(masks) <= expected_count:
        # still do one edge filtering, avoid mixing with edge when exporting
        masks = filter_masks_touching_border(masks, margin=3, rel_border_frac=0.002)
        return masks

    # ① edge removal (first do a coarse cleanup)
    masks = filter_masks_touching_border(masks, margin=3, rel_border_frac=0.002)

    # ② median & initial range
    areas = np.array([m["area"] for m in masks], dtype=float)
    mu = float(np.median(areas))
    low1, high1 = 0.80 * mu, 1.20 * mu
    low2, high2 = 0.75 * mu, 1.25 * mu  # backup放宽

    # ③ candidate scoring: the closer to the median, the better
    def _score(m): 
        return abs(m["area"] - mu)

    # pick in the tight range
    in1 = [m for m in masks if low1 <= m["area"] <= high1]
    in1.sort(key=_score)

    selected = in1[:expected_count]
    selected_ids = {id(m) for m in selected}   # ← use object identity tracking

    # if not enough, relax
    if len(selected) < expected_count:
        remain = [m for m in masks 
                  if id(m) not in selected_ids and low2 <= m["area"] <= high2]
        remain.sort(key=_score)
        need = expected_count - len(selected)
        selected.extend(remain[:need])
        selected_ids.update(id(m) for m in remain[:need])

    # if still not enough, fill from the whole by distance
    if len(selected) < expected_count:
        pool = [m for m in masks if id(m) not in selected_ids]
        pool.sort(key=_score)
        need = expected_count - len(selected)
        selected.extend(pool[:need])
        selected_ids.update(id(m) for m in pool[:need])

    # ④ deduplication (auto: keep the one closer to the median)
    before = len(selected)
    selected = deduplicate_masks(selected, iou_thresh=0.80, keep='auto')
    if len(selected) != before:
        print(f"[DEBUG] select_by_expected NMS {before}->{len(selected)}")

    # ⑤ if the number of deduplication is less, continue to fill from the remaining by distance to exact K
    if len(selected) < expected_count:
        selected_ids = {id(m) for m in selected}
        pool = [m for m in masks if id(m) not in selected_ids]
        pool.sort(key=_score)
        need = expected_count - len(selected)
        selected.extend(pool[:need])

    # ⑥ final truncation/ensure exact number
    selected = selected[:expected_count]

    print(f"[DEBUG] Selected {len(selected)}/{expected_count} (mu={mu:.1f})")
    return selected


def run_section_detection(image_path, model_type, apply_filtering, filter_strength,
                          expected_count=None, force_recompute=False,
                          enable_smoothing=True, smoothing_strength=0.005):
    """
    Wrapper for automatic_identification / DownsampledSAMDetector
    - NMS (deduplication) → edge contact filtering → area percentile filtering → contour closure check → optional expected_count → smoothing
    """
    checkpoint = get_sam_checkpoint(model_type)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache_file = f"{image_path}_masks.pkl"
    need_recompute = force_recompute or (not os.path.exists(cache_file))

    # A) need to re-segment → DownsampledSAMDetector (adaptive factor)
    if need_recompute:
        model_key_map = {
            "vit_b": "sam1_vit_b",
            "vit_l": "sam1_vit_l",
            "vit_h": "sam1_vit_h"
        }
        img0 = cv2.imread(image_path)
        H0, W0 = img0.shape[:2]
        long_edge = max(H0, W0)
        target = 3000
        factor = max(1, int(round(long_edge / target)))

        detector = DownsampledSAMDetector(
            model_key=model_key_map.get(model_type, "sam1_vit_b"),
            downsample_factor=factor
        )
        print(f"[DEBUG] DownsampledSAMDetector segmenting... (factor×{factor})")
        raw_masks = detector.detect_masks(
            image_path,
            min_area=800,
            max_area=50000
        )
        # uniform schema
        masks = []
        for m in raw_masks:
            seg = _ensure_seg(m).astype(bool)
            masks.append(_to_std_mask_dict(seg, score=float(m.get('score', 0.0))))

        with open(cache_file, "wb") as f:
            pickle.dump(masks, f)
        print(f"[DEBUG] cache saved → {cache_file}")

    # B) hit cache → directly read
    else:
        print(f"[DEBUG] read cache ← {cache_file}")
        with open(cache_file, "rb") as f:
            masks = pickle.load(f)

    print(f"[DEBUG] After SAM: {len(masks)} masks")
    if masks:
        areas = [m['area'] for m in masks]
        print(f"[DEBUG] Areas after SAM: min={np.min(areas)}, max={np.max(areas)}, median={np.median(areas)}")
        if SAVE_DEBUG_VIS:
            save_masks_visualization(image_path, masks, "debug_masks_after_sam.png")

    # —— NMS: prioritize the one "closer to the median area" (avoid keeping fragments/huge edge)  # CHANGED
    masks = deduplicate_masks(masks, iou_thresh=0.80, keep='auto')

    # —— NEW: edge contact filtering (remove wafer edge/large edge)        # NEW
    masks = filter_masks_touching_border(masks, margin=3, rel_border_frac=0.002)
    if masks and SAVE_DEBUG_VIS:
        save_masks_visualization(image_path, masks, "debug_masks_after_border.png")

    # area filtering (percentile P2..P98, double clipping; function body has been changed)         # CHANGED (implementation is placed in remove_small_masks)
    print(f"[DEBUG] Before area filtering: {len(masks)} masks")
    masks = remove_small_masks(masks)
    print(f"[DEBUG] After area filtering: {len(masks)} masks")
    if masks:
        areas = [m['area'] for m in masks]
        print(f"[DEBUG] Areas after area filtering: min={np.min(areas)}, max={np.max(areas)}, median={np.median(areas)}")
        if SAVE_DEBUG_VIS:
            save_masks_visualization(image_path, masks, "debug_masks_after_area.png")

    # contour closure check (keep your original logic)
    print(f"[DEBUG] Before contour filtering: {len(masks)} masks")
    filtered_masks = []
    for mask in masks:
        m = _ensure_seg(mask)
        contours = measure.find_contours(m.astype(np.uint8), 0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda x: x.shape[0])
        if contour.shape[0] < 3:
            continue
        contour_list = [[float(x), float(y)] for y, x in contour]
        start_point = np.array(contour_list[0])
        end_point = np.array(contour_list[-1])
        distance = np.linalg.norm(start_point - end_point)
        contour_length = len(contour_list)
        threshold = max(5.0, contour_length * 0.01)
        if distance <= threshold:
            filtered_masks.append(mask)
    print(f"[DEBUG] After contour filtering: {len(filtered_masks)} masks")
    if SAVE_DEBUG_VIS:
        save_masks_visualization(image_path, filtered_masks, "debug_masks_after_contour.png")
    masks = filtered_masks

    # more lenient (try when not enough)
    if len(masks) < 5 and filtered_masks:
        lenient_masks = []
        for mask in filtered_masks:
            m = _ensure_seg(mask)
            contours = measure.find_contours(m.astype(np.uint8), 0.5)
            if contours:
                contour = max(contours, key=lambda x: x.shape[0])
                if contour.shape[0] >= 10:
                    lenient_masks.append(mask)
        if len(lenient_masks) > len(masks):
            print(f"[DEBUG] Lenient filtering found {len(lenient_masks)} sections vs {len(masks)} strict")
            if SAVE_DEBUG_VIS:
                save_masks_visualization(image_path, lenient_masks, "debug_masks_after_lenient.png")
            masks = lenient_masks

    print(f"[DEBUG] Final mask count (pre-select): {len(masks)}")

    # optional number control: Top-K by "distance to median", and force to expected_count  # CHANGED
    if expected_count:
        masks = select_masks_by_expected_count(masks, expected_count)

    # boundary smoothing
    print(f"[DEBUG] Applying boundary smoothing...")
    masks = smooth_mask_boundaries(masks, smoothing_strength=smoothing_strength, enable_smoothing=enable_smoothing)
    if masks:
        areas = [m['area'] for m in masks]
        print(f"[DEBUG] After smoothing: min={np.min(areas)}, max={np.max(areas)}, median={np.median(areas)}")
        if SAVE_DEBUG_VIS:
            save_masks_visualization(image_path, masks, "debug_masks_after_smoothing.png")

    return masks


# === Helper utilities ===





def remove_small_masks(masks, area_ratio_thresh=0.05):
    """
    To keep compatibility, keep the parameter name, but actually use percentile clipping:
      lower bound = P2; upper bound = P98
    This can remove very small "come-in" points and also cut out extremely large abnormal blocks on the edge.
    """
    if not masks:
        return masks
    areas = np.array([m['area'] for m in masks], dtype=float)
    lo = np.percentile(areas, 2)
    hi = np.percentile(areas, 98)
    kept = [m for m in masks if lo <= m['area'] <= hi]
    if len(kept) != len(masks):
        print(f"[DEBUG] area percentile clip P2..P98 kept {len(kept)}/{len(masks)}")
    return kept


def smooth_mask_boundaries(masks, smoothing_strength=0.005, enable_smoothing=True):
    """smooth boundaries; write back segmentation/area/bbox"""
    if not enable_smoothing:
        return masks
        
    smoothed_masks = []
    for mask in masks:
        mask_uint8 = _ensure_seg(mask)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            smoothed_masks.append(mask)
            continue
        
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = float(smoothing_strength) * perimeter
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        h, w = mask_uint8.shape[:2]
        smoothed_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(smoothed_mask, [simplified_contour], 1)
        
        new_m = mask.copy()
        new_m['segmentation'] = smoothed_mask.astype(bool)
        new_m['area'] = int(smoothed_mask.sum())
        x, y, ww, hh = cv2.boundingRect(simplified_contour)
        new_m['bbox'] = [x, y, ww, hh]
        new_m['contour'] = simplified_contour.squeeze()
        smoothed_masks.append(new_m)
    
    print(f"[DEBUG] Smoothed {len(masks)} masks with strength={smoothing_strength}")
    return smoothed_masks

def segment_large_image_with_patches(np_img, model_type, device, patch_size=2048, points_per_side=32):
    """keep original function; fix bbox to [x,y,w,h] consistent"""
    from section_identification.section_detector import SectionDetector
    detector = SectionDetector(model_type=model_type, device=device)

    h, w = np_img.shape[:2]
    h_patches = (h + patch_size - 1) // patch_size
    w_patches = (w + patch_size - 1) // patch_size
    total_patches = h_patches * w_patches
    print(f"[DEBUG] Total patches to process: {total_patches} ({h_patches}×{w_patches})")
    sys.stdout.flush()

    raw_masks = []
    patch_idx = 0
    for i in range(h_patches):
        y_start = i * patch_size
        y_end = min((i + 1) * patch_size, h)
        for j in range(w_patches):
            x_start = j * patch_size
            x_end = min((j + 1) * patch_size, w)
            patch_idx += 1
            print(f"[DEBUG] Processing patch {patch_idx}/{total_patches}  (x:{x_start}-{x_end}, y:{y_start}-{y_end})")
            sys.stdout.flush()

            patch_img = np_img[y_start:y_end, x_start:x_end]
            patch_masks = detector.process_image(patch_img)

            for m in patch_masks:
                if m is not None:
                    m['contour'] = m['contour'] + np.array([x_start, y_start])
                    raw_masks.append(m)

    print("[DEBUG] All patches processed.")
    sys.stdout.flush()

    full_masks = []
    for m in raw_masks:
        contour = m['contour'].astype(np.int32)
        mask_bool = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_bool, [contour], 1)

        area = int(mask_bool.sum())
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = [x, y, bw, bh]  # FIX: 一致化

        full_masks.append({
            'segmentation': mask_bool.astype(bool),
            'area': area,
            'bbox': bbox,
            'score': float(m.get('score', 0.0))
        })

    detector.cleanup()
    print(f"[DEBUG] segment_large_image_with_patches produced {len(full_masks)} masks")
    return full_masks

def draw_mask_label(overlay_img, binary_mask_disp, label_id, scale=1.0):
    ys, xs = np.where(binary_mask_disp > 0)
    if xs.size == 0:
        return
    cx, cy = int(xs.mean()), int(ys.mean())
    half_w, half_h = int(24*scale), int(12*scale)
    cv2.rectangle(overlay_img, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), (160, 160, 160), -1)
    text = str(label_id)
    font_scale = 0.4*scale
    thickness = max(1, int(1.5*scale))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = cx - tw // 2
    text_y = cy + th // 2
    cv2.putText(overlay_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
