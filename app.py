import dash
# Standard Dash imports (no background callbacks)
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import numpy as np
import ast
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
from scipy.fft import fft, ifft
from numpy.linalg import lstsq
from PIL import Image
import time
import json
import plotly.io as pio
import matplotlib
matplotlib.use('Agg')  # Force non-GUI backend
import matplotlib.pyplot as plt
from dash.exceptions import PreventUpdate
from section_identification.section_detector import automatic_identification
from section_identification.interactive import run_sam_interactive
from section_identification.export import export_mask_coordinates
from modules.pages.section_counter import layout as section_counter_layout, register_section_counter_callbacks
from modules.pages.section_sequencing import sequencing_layout, register_sequencing_callbacks
import os
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QMessageBox, QProgressBar, QComboBox, QSpinBox,
                            QDoubleSpinBox, QCheckBox, QGroupBox, QRadioButton,
                            QButtonGroup, QTabWidget, QScrollArea, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSize
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import zscore
from skimage.measure import find_contours
from datetime import datetime
import traceback
import logging
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import torch
from segment_anything import sam_model_registry, SamPredictor
from modules.pages import section_order_overlap

# Custom CSS styles
app_css = {
    'custom-container': {
        'padding': '20px',
        'background-color': '#f8f9fa',
        'min-height': '100vh'
    },
    'sidebar': {
        'background-color': '#343a40',
        'padding': '20px',
        'height': '100vh',
        'position': 'fixed',
        'width': '250px'
    },
    'main-content': {
        'margin-left': '250px',
        'padding': '20px'
    },
    'nav-link': {
        'color': '#fff',
        'margin': '10px 0',
        'padding': '10px',
        'border-radius': '5px',
        'transition': 'all 0.3s'
    },
    'nav-link:hover': {
        'background-color': '#495057',
        'color': '#fff'
    },
    'nav-link.active': {
        'background-color': '#007bff',
        'color': '#fff'
    },
    'upload-button': {
        'width': '100%',
        'margin': '10px 0',
        'padding': '10px',
        'border-radius': '5px'
    },
    'graph-container': {
        'background-color': '#fff',
        'padding': '20px',
        'border-radius': '10px',
        'box-shadow': '0 0 10px rgba(0,0,0,0.1)'
    }
}

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        '/assets/custom.css'  # Add reference to custom CSS
    ],
    suppress_callback_exceptions=True,
)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Polygon Tools V0.3.0</title>
        {%favicon%}
        {%css%}
        <style>
            .custom-container { padding: 20px; background-color: #f8f9fa; min-height: 100vh; }
            .sidebar { background-color: #343a40; padding: 20px; height: 100vh; position: fixed; width: 250px; }
            .main-content { margin-left: 250px; padding: 20px; }
            .nav-link { color: #fff; margin: 10px 0; padding: 10px; border-radius: 5px; transition: all 0.3s; }
            .nav-link:hover { background-color: #495057; color: #fff; }
            .nav-link.active { background-color: #007bff; color: #fff; }
            .upload-button { width: 100%; margin: 10px 0; padding: 10px; border-radius: 5px; }
            .graph-container { background-color: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .status-text { font-size: 0.9em; margin-top: 5px; }
            .progress { height: 20px; margin-top: 10px; }
            .overlay-options { margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Wafer Tools", className="text-white sidebar-title"),
            dbc.Nav([
                dbc.NavLink("Section Counter", href="#", id="nav-section-counter", className="nav-link"),
                dbc.NavLink("Mask Unification & Alignment", href="#", id="nav-segmentation", className="nav-link"),
                dbc.NavLink("EM Imaging", href="#", id="nav-imaging", className="nav-link"),
                dbc.NavLink("Section Sequencing", href="#", id="nav-sequencing", className="nav-link"),
                dbc.NavLink("ORDER Visualization", href="#", id="nav-wafer-align", className="nav-link"),
            ], vertical=True, pills=True, className="sidebar-nav")
        ], width=2, className="sidebar"),
        dbc.Col([
            html.Div(id='page-content', className="main-content")
        ])
    ]),
    dcc.Store(id='intermediate-store'),
    dcc.Store(id='image-store'),
    dcc.Store(id='csv-name-store'),
    dcc.Store(id='deleted-polygons', data=[])
], fluid=True, className="custom-container")
alignment_layout = html.Div([
    html.H3("Mask Unification & Alignment", className="mb-4 text-primary"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=dbc.Button('Upload Image', color='primary', className='upload-button'),
                multiple=False
            ),
            html.Div(id='image-upload-status', className='status-text text-success')
        ], width=3),
        dbc.Col([
            dcc.Upload(
                id='upload-csv',
                children=dbc.Button('Upload CSV', color='secondary', className='upload-button'),
                multiple=False
            ),
            html.Div(id='csv-upload-status', className='status-text text-success')
        ], width=3),
        dbc.Col([
            dbc.Button('Unify All Masks', id='run-btn', color='success', className='upload-button'),
            dbc.Progress(id="progress-bar", value=0, striped=True, animated=True, className="progress")
        ], width=3),
        dbc.Col([
            dbc.Button("Confirm & Export", id="export-button", color="danger", className='upload-button'),
            dbc.Progress(id="export-progress-bar", value=0, striped=True, animated=True, className="progress"),
            html.Div(id='export-status', className='status-text text-info')
        ], width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='area-histogram', config={"displaylogo": False}),
                html.Div(id='area-warning', className='text-danger mt-2'),
                html.Div(id='manual-template-select-div'),
                dcc.Dropdown(id='manual-template-select', options=[], style={'display': 'none'})
            ], className="graph-container")
        ], width=4),
        # —— alignment_layout right column width=8 replace the whole column —— 
        dbc.Col([
            dbc.Row([
                # control panel (overlay + legend)
                dbc.Col([
                    # existing overlay options (unchanged; controlled by overlay-options-div)
                    html.Div([
                        html.Label("Overlay Options:", className="mb-2"),
                        dcc.Checklist(
                            id='overlay-options',
                            options=[
                                {'label': 'Show Masks', 'value': 'masks'},
                                {'label': 'Show Markers', 'value': 'markers'}
                            ],
                            value=['masks', 'markers'],
                            labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                        )
                    ], id='overlay-options-div', className="overlay-options", style={'display': 'none'}),

                    # new: legend panel (can be hidden/restored)    # CHANGED
                    html.Div([
                        html.Label("Legend (select to hide):", className="mb-1"),
                        dcc.Dropdown(
                            id='legend-dropdown',
                            options=[],          # filled in callback
                            value=[],            # selected ids to hide
                            multi=True,
                            placeholder="Select sections to hide…",
                        ),
                        html.Div(id='legend-status', className='text-muted mt-1', style={'fontSize': '0.9em'}),
                        dbc.Button("Reset", id="legend-reset-btn", color="secondary", size="sm", className="mt-2"),
                    ], className="overlay-options")
                ], width=2, style={'minWidth': '180px', 'maxWidth': '220px', 'paddingRight': '0'}),

                # 右：大图
                dbc.Col([
                    html.Div([
                        dcc.Graph(id='polygon-graph', config={"displaylogo": False, "scrollZoom": True})
                    ], className="graph-container")
                ], width=10),
            ])
        ], width=8)
    ], className="mb-4"),
])
placeholder_layout = lambda title: html.Div([
    html.H3(f"{title} (Coming Soon)", className="text-muted text-center my-5"),
    html.Div([
        html.I(className="fas fa-tools fa-3x text-muted mb-3"),
        html.P("This feature is under development. Please check back later!", className="text-muted")
    ], className="text-center")
], className="graph-container")
@app.callback(Output('page-content', 'children'),
              [Input('nav-section-counter', 'n_clicks'),
               Input('nav-segmentation', 'n_clicks'),
               Input('nav-imaging', 'n_clicks'),
               Input('nav-sequencing', 'n_clicks'),
               Input('nav-wafer-align', 'n_clicks')])
def render_page(n1, n2, n3, n4, n5):
    ctx = dash.callback_context
    if not ctx.triggered:
        return section_counter_layout
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'nav-section-counter':
            return section_counter_layout
        elif button_id == 'nav-segmentation':
            return alignment_layout
        elif button_id == 'nav-imaging':
            return html.Div([
                html.H3("EM Imaging", className="mb-4 text-primary"),
                html.P("EM Imaging functionality coming soon...")
            ])
        elif button_id == 'nav-sequencing':
            return sequencing_layout
        elif button_id == 'nav-wafer-align':
            return section_order_overlap.overlap_layout
        else:
            return section_counter_layout
def resample_polygon(poly, n_points=100):
    poly = np.asarray(poly)
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    tck, _ = splprep([poly[:, 0], poly[:, 1]], s=0, per=True)
    u = np.linspace(0, 1, n_points, endpoint=False)
    x, y = splev(u, tck)
    return np.stack([x, y], axis=1)
def compute_distance_profile(poly):
    center = poly.mean(axis=0)
    return np.linalg.norm(poly - center, axis=1)
def circular_shift(arr, shift):
    return np.concatenate([arr[shift:], arr[:shift]], axis=0)
def align_polygon_by_circular_correlation(poly, template_profile):
    profile = compute_distance_profile(poly)
    fft1 = fft(profile)
    fft2 = fft(template_profile)
    corr = ifft(fft1 * np.conj(fft2)).real
    shift = np.argmax(corr)
    return circular_shift(poly, shift), shift
def compute_affine_transform(src_pts, dst_pts):
    N = src_pts.shape[0]
    X = np.hstack([src_pts, np.ones((N, 1))])
    Y = dst_pts
    coeffs, _, _, _ = lstsq(X, Y, rcond=None)
    A = coeffs[:2, :].T
    b = coeffs[2, :]
    return A, b
def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@app.callback(
    Output('area-histogram', 'figure'),
    Output('area-warning', 'children'),
    Output('manual-template-select-div', 'children'),
    Output('manual-template-select', 'value'),
    Input('upload-csv', 'contents'),
    Input('upload-image', 'contents'),
    State('upload-csv', 'filename'),
    State('manual-template-select', 'value')
)
def analyze_area_and_template(csv_contents, image_contents, csv_filename, manual_template_idx):
    # no CSV: draw placeholder  
    if not csv_contents:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "Please upload CSV to see area distribution",
                'xref': "paper", 'yref': "paper",
                'x': 0.5, 'y': 0.5,
                'xanchor': 'center', 'yanchor': 'middle',
                'showarrow': False,
                'font': {'size': 14},
                'align': 'center'
            }],
            height=350,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        return fig, "", None, None

    # read CSV (only keep section_*)
    content_type, content_string = csv_contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    df = df[df['id'].str.startswith('section_')].copy()

    # polygon → resample → area
    polys = []
    for _, row in df.iterrows():
        coords = ast.literal_eval(row['contour_coordinates'])
        if isinstance(coords[0][0], list):
            coords = coords[0]
        polys.append(resample_polygon(coords, n_points=100))

    if len(polys) == 0:
        fig = go.Figure()
        fig.update_layout(height=350, template="plotly_white")
        return fig, "No valid polygons in CSV.", None, None

    areas = np.array([polygon_area(p) for p in polys], dtype=float)
    mean_area = float(np.mean(areas))
    band_lo, band_hi = mean_area * 0.9, mean_area * 1.1
    similar_mask = (areas >= band_lo) & (areas <= band_hi)
    similar_ratio = float(np.mean(similar_mask))
    similar_idx = np.flatnonzero(similar_mask)

    # histogram + green similar band
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=areas,
        nbinsx=20,
        marker=dict(color='rgba(0,123,255,0.7)', line=dict(color='rgba(0,0,0,0.8)', width=1)),
        opacity=0.85,
        name='Polygon Areas',
        hovertemplate='Area: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    fig.add_vrect(x0=band_lo, x1=band_hi, fillcolor="rgba(40,167,69,0.20)", line_width=0)
    fig.update_layout(
        title="Polygon Area Distribution",
        xaxis_title="Area",
        yaxis_title="Count",
        bargap=0.1,
        template="plotly_white",
        height=350
    )

    # select template: auto (similarity>80%) or manual (show dropdown)
    warning = ""
    manual_div = None

    if similar_ratio > 0.8 and len(similar_idx) > 0:
        # in "similar set", pick the index of the one closest to the median of the similar set
        sim_areas = areas[similar_idx]
        med_sim = float(np.median(sim_areas))
        picked = int(similar_idx[np.argmin(np.abs(sim_areas - med_sim))])

        picked_id = df.iloc[picked]['id']
        warning = (
            f"Congrats! {similar_ratio*100:.1f}% of polygons have similar area. "
            f"Auto-picked template: {picked_id} (area≈{areas[picked]:.2f}, "
            f"band {band_lo:.2f} ~ {band_hi:.2f})."
        )
        # don't create second dropdown; directly set the value of "hidden manual-template-select" to picked
        return fig, warning, None, picked

    # need to select manually: show a visible dropdown (still reuse the same id)
    options = [{'label': f"{row['id']} (Area: {areas[i]:.2f})", 'value': int(i)} for i, row in df.iterrows()]
    default_val = 0 if manual_template_idx is None else int(manual_template_idx)
    manual_div = html.Div([
        html.Label("Manually select template polygon:"),
        dcc.Dropdown(
            id='manual-template-select',
            options=options,
            value=default_val,
            clearable=False,
            style={"width": "60%"}
        )
    ], className="mb-2")

    warning = f"⚠️ Only {similar_ratio*100:.1f}% of polygons have similar area, please select the template manually!"
    return fig, warning, manual_div, default_val


# use intermediate data to build legend options, and sync with deleted-polygons initially
@app.callback(
    Output('legend-dropdown', 'options'),
    Input('intermediate-store', 'data'),
    prevent_initial_call=True
)
def build_legend_options(json_data):
    if not json_data:
        raise PreventUpdate

    df = pd.read_json(io.StringIO(json_data))
    ids = [str(x) for x in df['id'].tolist()]
    options = [{'label': s, 'value': s} for s in ids]
    return options



@app.callback(
    Output('intermediate-store', 'data'),
    Output('csv-name-store', 'data'),
    Output('progress-bar', 'value'),
    Input('run-btn', 'n_clicks'),
    State('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('manual-template-select', 'value'),
    prevent_initial_call=True
)
def process_alignment(n_clicks, csv_contents, csv_filename, manual_template_idx):
    try:
        if not csv_contents:
            raise PreventUpdate

        # read CSV (only keep section_*)
        content_type, content_string = csv_contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = df[df['id'].str.startswith('section_')].copy()

        # polygon → resample
        polygons = []
        for _, row in df.iterrows():
            coords = ast.literal_eval(row['contour_coordinates'])
            if isinstance(coords[0][0], list):
                coords = coords[0]
            polygons.append(resample_polygon(coords, n_points=100))

        if len(polygons) == 0:
            return dash.no_update, dash.no_update, dash.no_update

        # ① determine template index
        if manual_template_idx is not None:
            template_idx = int(manual_template_idx)
        else:
            # same as visualization: when >80% similar, pick the median representative from the similar set; otherwise fallback to 0
            areas = np.array([polygon_area(p) for p in polygons], dtype=float)
            mean_area = float(np.mean(areas))
            band_lo, band_hi = mean_area * 0.9, mean_area * 1.1
            similar_mask = (areas >= band_lo) & (areas <= band_hi)
            if np.mean(similar_mask) > 0.8 and np.any(similar_mask):
                sim_idx = np.flatnonzero(similar_mask)
                sim_areas = areas[sim_idx]
                med_sim = float(np.median(sim_areas))
                template_idx = int(sim_idx[np.argmin(np.abs(sim_areas - med_sim))])
            else:
                template_idx = 0

        template_poly = polygons[template_idx]

        # ② align with template (keep your original logic)
        template_profile = compute_distance_profile(template_poly)

        aligned_polys = []
        for poly in polygons:
            aligned_poly, _ = align_polygon_by_circular_correlation(poly, template_profile)
            aligned_polys.append(aligned_poly)

        warped_polys, centers, angles = [], [], []
        for tgt_poly in aligned_polys:
            A, b = compute_affine_transform(template_poly, tgt_poly)
            warped = (template_poly @ A.T) + b
            warped_polys.append(warped.tolist())

            c_x, c_y = warped.mean(axis=0)
            centers.append((float(c_x), float(c_y)))

            vx = warped[0, 0] - c_x
            vy = warped[0, 1] - c_y
            angle_ccw = (np.degrees(np.arctan2(vy, vx)) + 360) % 360
            angles.append(float(angle_ccw))

        json_data = pd.DataFrame({
            'id': df['id'].values,
            'warped_coordinates': [str(p) for p in warped_polys],
            'center_x': [c[0] for c in centers],
            'center_y': [c[1] for c in centers],
            'angle_deg': angles
        }).to_json()

        return json_data, csv_filename, 100

    except Exception:
        # don't overwrite existing UI on error
        return dash.no_update, dash.no_update, dash.no_update

    
@app.callback(
    Output('polygon-graph', 'figure'),
    [Input('upload-image', 'contents'),
     Input('intermediate-store', 'data'),
     Input('deleted-polygons', 'data'),
     Input('overlay-options', 'value')],
    prevent_initial_call=True
)
def plot_image_and_polygons(image_content, json_data, deleted_ids, overlay_options):
    if not image_content:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{
                'text': "Please upload image and CSV",
                'xref': "paper", 'yref': "paper",
                'showarrow': False,
                'font': {'size': 20}
            }],
            height=800,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig

    deleted_ids = deleted_ids or []  # ← prevent None
    content_type, content_string = image_content.split(',')
    img_bytes = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(img_bytes))
    orig_width, orig_height = img.size

    # generate thumbnail ratio
    max_dim = 1000
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), resample)
    thumb_width, thumb_height = img.size
    scale_x = thumb_width / orig_width
    scale_y = thumb_height / orig_height
    img_array = np.array(img)

    fig = go.Figure()
    fig.add_trace(go.Image(z=img_array))

    if json_data:
        df = pd.read_json(io.StringIO(json_data))
        for _, row in df.iterrows():
            pid = row['id']
            # don't show deleted (not selected)
            if pid in deleted_ids:
                continue

            try:
                poly = np.array(ast.literal_eval(row['warped_coordinates']))
                if not np.isfinite(poly).all():
                    continue

                # scale polygon coordinates proportionally
                poly_scaled = poly.copy()
                poly_scaled[:, 0] = poly[:, 0] * scale_x
                poly_scaled[:, 1] = poly[:, 1] * scale_y

                if 'masks' in overlay_options:
                    fig.add_trace(go.Scatter(
                        x=poly_scaled[:, 0],
                        y=poly_scaled[:, 1],
                        mode='lines',
                        name=pid,
                        line=dict(width=2, color='rgba(0,0,255,0.6)'),
                        text=[pid] * len(poly_scaled),                # fallback
                        customdata=[pid] * len(poly_scaled),          # ★ carry id
                        hovertemplate="id=%{customdata}<extra></extra>",
                        showlegend=False
                    ))

                if 'markers' in overlay_options:
                    # 质心
                    cx = row['center_x'] if 'center_x' in row else float(poly_scaled[:, 0].mean())
                    cy = row['center_y'] if 'center_y' in row else float(poly_scaled[:, 1].mean())
                    cx *= scale_x if 'center_x' in row else 1.0
                    cy *= scale_y if 'center_y' in row else 1.0

                    fig.add_trace(go.Scatter(
                        x=[cx], y=[cy], mode='markers',
                        marker=dict(color='red', size=6, symbol='x'),
                        text=[pid],                                 # fallback
                        customdata=[pid],                           # ★ carry id        
                        hovertemplate="id=%{customdata}<extra></extra>",
                        name=f"{pid}-center",
                        showlegend=False
                    ))

                    # 首点
                    fx, fy = poly_scaled[0, 0], poly_scaled[0, 1]
                    angle_ccw = row.get('angle_deg', 0.0)
                    angle_text = f"{pid} ({angle_ccw:.1f}° CCW)"
                    fig.add_trace(go.Scatter(
                        x=[fx], y=[fy], mode='markers',
                        marker=dict(color='lime', size=6, symbol='circle'),
                        text=[angle_text],                          # readable text
                        customdata=[pid],                           # ★ carry id
                        hovertemplate="id=%{customdata}<extra></extra>",
                        name=f"{pid}-first",
                        showlegend=False
                    ))

                    # 连接线
                    fig.add_trace(go.Scatter(
                        x=[cx, fx], y=[cy, fy], mode='lines',
                        line=dict(color='red', width=2),
                        hoverinfo='skip',
                        customdata=[pid, pid],                      # ★ carry id
                        name=f"{pid}-link",
                        showlegend=False
                    ))
            except Exception:
                continue

    fig.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(autorange='reversed'),
        uirevision=hash(frozenset(overlay_options))  # keep zoom/view
    )
    return fig

@app.callback(
    Output('deleted-polygons', 'data'),
    Output('legend-dropdown', 'value'),
    Output('legend-status', 'children'),
    Input('polygon-graph', 'clickData'),
    Input('legend-dropdown', 'value'),
    Input('legend-reset-btn', 'n_clicks'),
    Input('intermediate-store', 'data'),
    State('deleted-polygons', 'data'),
    State('legend-dropdown', 'options'),
    prevent_initial_call=True
)
def update_deleted(click_data, legend_hidden, reset_n, json_data, deleted, options):
    deleted = (deleted or []).copy()
    trig = ctx.triggered_id

    def status_from(_deleted, _options):
        total = len(_options or [])
        hidden = len([x for x in (_deleted or []) if any(opt['value'] == x for opt in (_options or []))])
        shown = total - hidden
        return f"{shown}/{total} shown — selected are hidden."

    # 1) new data loaded: mirror current deleted to legend.value, and refresh status
    if trig == 'intermediate-store':
        value = [x for x in deleted if any(opt['value'] == x for opt in (options or []))]
        return deleted, value, status_from(deleted, options)

    # 2) Reset: show all
    if trig == 'legend-reset-btn':
        return [], [], status_from([], options)

    # 3) Legend dropdown: value is the set to hide → directly overwrite
    if trig == 'legend-dropdown':
        norm = [x for x in (legend_hidden or []) if any(opt['value'] == x for opt in (options or []))]
        return norm, norm, status_from(norm, options)

    # 4) click on graph: switch single id
    if trig == 'polygon-graph' and click_data and 'points' in click_data and click_data['points']:
        p = click_data['points'][0]
        pid = None

        cd = p.get('customdata')
        if isinstance(cd, (list, tuple)):
            pid = cd[0] if cd else None
        elif cd:
            pid = cd

        if not pid:
            txt = p.get('text', '')
            if isinstance(txt, str) and txt:
                pid = txt.split(' (', 1)[0]

        if not pid:
            value = [x for x in deleted if any(opt['value'] == x for opt in (options or []))]
            return deleted, value, status_from(deleted, options)

        if pid in deleted:
            deleted = [x for x in deleted if x != pid]  # restore display
        else:
            deleted = deleted + [pid]                    # hide

        value = [x for x in deleted if any(opt['value'] == x for opt in (options or []))]
        return deleted, value, status_from(deleted, options)

    # 兜底：不改变
    value = [x for x in deleted if any(opt['value'] == x for opt in (options or []))]
    return deleted, value, status_from(deleted, options)



import matplotlib.pyplot as plt
@app.callback(
    Output("export-status", "children"),
    Output("export-progress-bar", "value"),
    Input("export-button", "n_clicks"),
    State("intermediate-store", "data"),
    State("upload-image", "contents"),
    State("deleted-polygons", "data"),
    State("overlay-options", "value"),
    prevent_initial_call=True
)
def export_final(n_clicks, json_data, image_content, deleted_ids, overlay_options):
    try:
        deleted_ids = deleted_ids or [] 
        df = pd.read_json(io.StringIO(json_data))
        df_filtered = df[~df['id'].isin(deleted_ids)].copy()

        # Build required export format
        df_export = df_filtered.copy()
        # Ensure necessary columns exist
        if 'warped_coordinates' in df_export.columns and 'contour_coordinates' not in df_export.columns:
            df_export['contour_coordinates'] = df_export['warped_coordinates']
        if 'center_x' in df_export.columns and 'center_coordinates' not in df_export.columns:
            df_export['center_coordinates'] = df_export.apply(lambda r: [r['center_x'], r['center_y']], axis=1)
        df_export['type'] = 'new_mask'
        df_export['rotation'] = df_export['angle_deg']

        # Final CSV columns order
        cols = ['id', 'type', 'contour_coordinates', 'center_coordinates', 'rotation']
        ENABLE_LEGACY_RESULTS = bool(int(os.getenv('WAFER_ENABLE_LEGACY_RESULTS', '0')))
        # unified result directory
        try:
            from modules.common.paths import get_run_dir
            run_dir = get_run_dir('mask_unification')
        except Exception:
            # fallback: create a basic timestamped dir if helper unavailable
            import datetime
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join('results', 'mask_unification', ts)
            os.makedirs(run_dir, exist_ok=True)

        # save CSV primarily to unified directory
        unified_csv_path = os.path.join(run_dir, 'aligned_polygons.csv')
        df_export[cols].to_csv(unified_csv_path, index=False)
        # optionally also write legacy copy
        if ENABLE_LEGACY_RESULTS:
            if not os.path.exists('Result_masking'):
                os.makedirs('Result_masking')
            df_export[cols].to_csv(os.path.join('Result_masking', 'aligned_polygons.csv'), index=False)

        # Image export
        if image_content:
            _, content_string = image_content.split(',')
            img_bytes = base64.b64decode(content_string)
            img_array = np.array(Image.open(io.BytesIO(img_bytes)))

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img_array, cmap='gray')

            for _, row in df_filtered.iterrows():
                poly = np.array(ast.literal_eval(row['warped_coordinates']))
                if 'masks' in overlay_options:
                    ax.plot(poly[:, 0], poly[:, 1], linewidth=1.5, color='navy')  # deep blue edge
                if 'markers' in overlay_options:
                    # centroid and line
                    if 'center_x' in row and 'center_y' in row:
                        cx, cy = row['center_x'], row['center_y']
                    else:
                        cx, cy = poly[:,0].mean(), poly[:,1].mean()
                    ax.plot(cx, cy, marker='x', color='red')
                    ax.plot(poly[0,0], poly[0,1], marker='o', markerfacecolor='lime', markeredgecolor='green')
                    ax.plot([cx, poly[0,0]], [cy, poly[0,1]], color='red', linewidth=2)

            ax.axis('off')
            plt.tight_layout()
            # save to unified directory first
            plt.savefig(os.path.join(run_dir, 'filtered_output.png'), dpi=300, bbox_inches='tight')
            # optionally legacy copy
            if ENABLE_LEGACY_RESULTS:
                plt.savefig(os.path.join('Result_masking', 'filtered_output.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # write meta in unified dir
        try:
            from modules.common.io import save_meta
            save_meta(run_dir, {
                'module': 'mask_unification',
                'rows': int(len(df_export))
            })
        except Exception:
            pass

        unified_msg = f"results/mask_unification/{os.path.basename(run_dir)}"
        if ENABLE_LEGACY_RESULTS:
            return f"✅ Exported to {unified_msg} (legacy copy also written)", 100
        else:
            return f"✅ Exported to {unified_msg}", 100
    except Exception as e:
        return f"❌ Export failed: {str(e)}", 0
@app.callback(
    Output('csv-upload-status', 'children'),
    [Input('upload-csv', 'filename'),
     Input('csv-name-store', 'data')],
    prevent_initial_call=True
)
def unified_csv_status(uploaded_filename, stored_filename):
    trigger = ctx.triggered_id
    if trigger == 'upload-csv' and uploaded_filename:
        return f":white_check_mark: CSV uploaded: {uploaded_filename}"
    elif trigger == 'csv-name-store' and stored_filename:
        return f":white_check_mark: CSV loaded: {stored_filename}"
    return ""
@app.callback(
    Output('image-upload-status', 'children'),
    Input('upload-image', 'filename')
)
def update_image_status(filename):
    if filename:
        return f":white_check_mark: Image uploaded: {filename}"
    return ""
@app.callback(
    Output('overlay-options-div', 'style'),
    Input('intermediate-store', 'data'),
    prevent_initial_call=True
)
def show_overlay_options_style(intermediate_data):
    if intermediate_data:
        return {'display': 'block'}
    return {'display': 'none'}
@app.callback(
    [Output('nav-section-counter', 'active'),
     Output('nav-segmentation', 'active'),
     Output('nav-imaging', 'active'),
     Output('nav-sequencing', 'active'),
     Output('nav-wafer-align', 'active')],
    [Input('nav-section-counter', 'n_clicks'),
     Input('nav-segmentation', 'n_clicks'),
     Input('nav-imaging', 'n_clicks'),
     Input('nav-sequencing', 'n_clicks'),
     Input('nav-wafer-align', 'n_clicks')],
    prevent_initial_call=True
)
def update_nav_active(n1, n2, n3, n4, n5):
    triggered = ctx.triggered_id
    return [
        triggered == 'nav-section-counter',
        triggered == 'nav-segmentation',
        triggered == 'nav-imaging',
        triggered == 'nav-sequencing',
        triggered == 'nav-wafer-align'
    ]

# add new alignment function
def calculate_signature(contour):
    """Calculate signature for a contour"""
    # calculate centroid of the contour
    centroid = np.mean(contour, axis=0)
    
    # calculate distance from each point to the centroid
    distances = np.sqrt(np.sum((contour - centroid) ** 2, axis=1))
    
    # calculate signature using FFT
    signature = np.abs(fft(distances))
    
    # normalize
    signature = signature / np.max(signature)
    
    return signature

def circular_cross_correlation(sig1, sig2):
    """Calculate circular cross-correlation between two signatures"""
    # calculate circular cross-correlation using FFT
    fft1 = fft(sig1)
    fft2 = fft(sig2)
    correlation = np.real(ifft(fft1 * np.conj(fft2)))
    
    # normalize
    correlation = correlation / np.max(np.abs(correlation))
    
    return correlation

def estimate_rigid(contour1, contour2, shift):
    """Estimate rigid transformation between two contours"""
    # create rotation matrix
    angle = 2 * np.pi * shift / len(contour1)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # calculate centroid
    centroid1 = np.mean(contour1, axis=0)
    centroid2 = np.mean(contour2, axis=0)
    
    # calculate translation vector
    translation = centroid2 - np.dot(rotation_matrix, centroid1)
    
    return rotation_matrix, translation

def resample_contour(contour, n_points=100):
    """Resample contour to have n_points"""
    # calculate perimeter of the contour
    perimeter = np.sum(np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1)))
    
    # calculate target distance between each point
    target_distance = perimeter / n_points
    
    # create new contour points
    new_contour = []
    current_point = contour[0]
    new_contour.append(current_point)
    
    # iterate over original contour points
    remaining_distance = 0
    for i in range(1, len(contour)):
        # calculate distance to next point
        segment = contour[i] - contour[i-1]
        segment_length = np.sqrt(np.sum(segment ** 2))
        
        # if remaining distance plus current segment length is greater than target distance
        if remaining_distance + segment_length >= target_distance:
            # calculate number of points to insert
            n_insert = int((remaining_distance + segment_length) / target_distance)
            
            # insert points
            for j in range(n_insert):
                t = (target_distance - remaining_distance + j * target_distance) / segment_length
                new_point = contour[i-1] + t * segment
                new_contour.append(new_point)
            
            # update remaining distance
            remaining_distance = (remaining_distance + segment_length) % target_distance
        else:
            remaining_distance += segment_length
    
    return np.array(new_contour)

def detect_alignment_outliers(correlation_matrix, threshold_factor=2.0):
    """Detect sections with poor alignment quality"""
    # calculate alignment cost for each contour
    costs = 1 - np.max(correlation_matrix, axis=1)
    
    # detect outliers using Z-score
    z_scores = zscore(costs)
    outliers = np.where(np.abs(z_scores) > threshold_factor)[0]
    
    # calculate outlier scores
    outlier_scores = np.abs(z_scores[outliers])
    
    return outliers, outlier_scores, costs

def create_template_bounding_box(contour, margin=10):
    """Create bounding box for template contour"""
    # calculate bounding box of the contour
    min_x = np.min(contour[:, 0])
    max_x = np.max(contour[:, 0])
    min_y = np.min(contour[:, 1])
    max_y = np.max(contour[:, 1])
    
    # add margin
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # create four corners of the bounding box
    bbox = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])
    
    return bbox

def process_csv(self, csv_path):
    """Process CSV file with new alignment logic"""
    try:
        # read CSV file
        df = pd.read_csv(csv_path)
        
        # create output directory
        output_dir = 'contour_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # extract contours
        contours = []
        for idx, row in df.iterrows():
            if str(row['id']).startswith('section_'):
                try:
                    coords = eval(row['contour_coordinates'])
                    if len(coords) > 2:  # 确保至少有三个点
                        contours.append(np.array(coords))
                except:
                    continue
        
        if not contours:
            raise ValueError("No valid contours found in the CSV file")
        
        # resample contours to uniform number of points
        n_points = 100
        resampled_contours = [resample_contour(contour, n_points) for contour in contours]
        
        # calculate signature for each contour
        signatures = [calculate_signature(contour) for contour in resampled_contours]
        
        # calculate correlation matrix
        n_contours = len(resampled_contours)
        correlation_matrix = np.zeros((n_contours, n_contours))
        shift_matrix = np.zeros((n_contours, n_contours))
        
        for i in range(n_contours):
            for j in range(n_contours):
                if i != j:
                    correlation = circular_cross_correlation(signatures[i], signatures[j])
                    correlation_matrix[i, j] = np.max(correlation)
                    shift_matrix[i, j] = np.argmax(correlation)
        
        # detect outliers
        outliers, outlier_scores, costs = detect_alignment_outliers(correlation_matrix)
        
        # select template contour (use the contour with the lowest cost)
        template_idx = np.argmin(costs)
        template_contour = resampled_contours[template_idx]
        template_bbox = create_template_bounding_box(template_contour)
        
        # calculate transformation for each contour
        transformations = []
        for i in range(n_contours):
            if i != template_idx:
                shift = shift_matrix[i, template_idx]
                rotation_matrix, translation = estimate_rigid(
                    resampled_contours[i], 
                    template_contour, 
                    shift
                )
                transformations.append({
                    'rotation_matrix': rotation_matrix.tolist(),
                    'translation': translation.tolist()
                })
            else:
                transformations.append({
                    'rotation_matrix': np.eye(2).tolist(),
                    'translation': [0, 0]
                })
        
        # save results
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_aligned.csv")
        
        # create result DataFrame
        result_data = []
        for i, (contour, transform) in enumerate(zip(resampled_contours, transformations)):
            result_data.append({
                'id': f'contour_{i}',
                'contour_coordinates': contour.tolist(),
                'rotation_matrix': transform['rotation_matrix'],
                'translation': transform['translation'],
                'is_outlier': i in outliers,
                'alignment_cost': float(costs[i])
            })
        
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_file, index=False)
        
        # create visualization
        plt.figure(figsize=(12, 8))
        
        # plot all contours
        for i, contour in enumerate(resampled_contours):
            color = 'red' if i in outliers else 'blue'
            plt.plot(contour[:, 0], contour[:, 1], color=color, alpha=0.5)
        
        # plot template contour and bounding box
        plt.plot(template_contour[:, 0], template_contour[:, 1], 'g-', linewidth=2)
        plt.plot(template_bbox[:, 0], template_bbox[:, 1], 'g--')
        
        plt.title('Aligned Contours with Template')
        plt.axis('equal')
        
        # save visualization
        vis_file = os.path.join(output_dir, f"{base_name}_alignment_visualization.png")
        plt.savefig(vis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file, vis_file
        
    except Exception as e:
        logging.error(f"Error in process_csv: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def process_csv_file(self, csv_path):
    """Process CSV file with new alignment logic"""
    try:
        # call new process_csv function
        output_file, vis_file = self.process_csv(csv_path)
        
        # update UI display result
        self.result_label.setText(f"Processing completed! Results saved to:\n{output_file}\nVisualization saved to:\n{vis_file}")
        
        # display visualization result
        if os.path.exists(vis_file):
            self.display_image(vis_file)
        
        return True
        
    except Exception as e:
        self.result_label.setText(f"Processing failed: {str(e)}")
        logging.error(f"Error in process_csv_file: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def create_wafer_align_tab(self):
    """Create Wafer Align tab"""
    tab = QWidget()
    layout = QVBoxLayout()
    
    # add model selection component
    model_group = QGroupBox("SAM Model Selection")
    model_layout = QHBoxLayout()
    
    self.model_combo = QComboBox()
    self.model_combo.addItems([
        "vit_h (Heavy - Best Quality)",
        "vit_l (Light - Balanced)",
        "vit_b (Lighter - Faster)"
    ])
    self.model_combo.setCurrentIndex(1)  # default to vit_l
    self.model_combo.currentIndexChanged.connect(self.initialize_sam_model)  # add model switch event
    
    model_layout.addWidget(QLabel("Model:"))
    model_layout.addWidget(self.model_combo)
    model_group.setLayout(model_layout)
    layout.addWidget(model_group)
    
    # add image processing options
    process_group = QGroupBox("Image Processing Options")
    process_layout = QVBoxLayout()
    
    # add downsampling option
    self.downsample_check = QCheckBox("Enable Downsampling for Large Images")
    self.downsample_check.setChecked(True)
    process_layout.addWidget(self.downsample_check)
    
    # add downsampling ratio selection
    downsample_layout = QHBoxLayout()
    downsample_layout.addWidget(QLabel("Downsample Ratio:"))
    self.downsample_ratio = QDoubleSpinBox()
    self.downsample_ratio.setRange(0.1, 1.0)
    self.downsample_ratio.setValue(0.5)
    self.downsample_ratio.setSingleStep(0.1)
    downsample_layout.addWidget(self.downsample_ratio)
    process_layout.addLayout(downsample_layout)
    
    # add patch processing option
    self.patch_check = QCheckBox("Enable Patch Processing")
    self.patch_check.setChecked(True)
    process_layout.addWidget(self.patch_check)
    
    # add patch size selection
    patch_layout = QHBoxLayout()
    patch_layout.addWidget(QLabel("Patch Size:"))
    self.patch_size = QSpinBox()
    self.patch_size.setRange(512, 2048)
    self.patch_size.setValue(1024)
    self.patch_size.setSingleStep(256)
    patch_layout.addWidget(self.patch_size)
    process_layout.addLayout(patch_layout)
    
    process_group.setLayout(process_layout)
    layout.addWidget(process_group)
    
    # add buttons
    button_layout = QHBoxLayout()
    
    self.load_image_button = QPushButton("Load Image")
    self.load_image_button.clicked.connect(self.load_image)
    button_layout.addWidget(self.load_image_button)
    
    self.run_detection_button = QPushButton("Run Auto Detection")
    self.run_detection_button.clicked.connect(self.run_auto_detection)
    button_layout.addWidget(self.run_detection_button)
    
    self.save_csv_button = QPushButton("Save CSV")
    self.save_csv_button.clicked.connect(self.save_csv)
    button_layout.addWidget(self.save_csv_button)
    
    layout.addLayout(button_layout)
    
    # add image display area
    self.image_label = QLabel()
    self.image_label.setAlignment(Qt.AlignCenter)
    self.image_label.setMinimumSize(800, 600)
    self.image_label.setStyleSheet("border: 1px solid #cccccc;")
    layout.addWidget(self.image_label)
    
    # add result label
    self.result_label = QLabel()
    self.result_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(self.result_label)
    
    tab.setLayout(layout)
    return tab

def __init__(self):
    super().__init__()
    self.setWindowTitle("Wafer Section Tool")
    self.setGeometry(100, 100, 1200, 800)
    
    # initialize variables
    self.current_image = None
    self.current_image_path = None
    self.masks = []
    self.current_mask_index = -1
    self.interactive_helper = None
    
    # create main layout
    main_layout = QVBoxLayout()
    
    # create tabs
    self.tab_widget = QTabWidget()
    self.tab_widget.addTab(self.create_wafer_align_tab(), "Wafer Align")
    self.tab_widget.addTab(self.create_section_counter_tab(), "Section Counter")
    main_layout.addWidget(self.tab_widget)
    
    self.setLayout(main_layout)
    
    # initialize SAM model
    self.initialize_sam_model()

def initialize_sam_model(self):
    """Initialize SAM model based on selected type"""
    try:
        model_type = self.model_combo.currentText().split()[0]
        
        # if helper is already initialized and type is the same, return
        if self.interactive_helper is not None and self.interactive_helper.detector.model_type == model_type:
            return
        
        # clean up previous helper
        if self.interactive_helper is not None:
            self.interactive_helper.cleanup()
        
        # initialize new helper
        self.interactive_helper = InteractiveHelper(model_type)
        print(f"Initialized SAM model: {model_type}")
        
    except Exception as e:
        logging.error(f"Error initializing SAM model: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def load_image(self):
    """Load image and initialize processing"""
    try:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            # load image
            self.current_image = self.interactive_helper.load_image(file_path)
            self.current_image_path = file_path
            
            # display image
            self.display_image(file_path)
            
            # update UI
            self.result_label.setText("Image loaded successfully")
            
    except Exception as e:
        self.result_label.setText(f"Loading failed: {str(e)}")
        logging.error(f"Error in load_image: {str(e)}")
        logging.error(traceback.format_exc())

def run_auto_detection(self):
    """Run automatic detection with SAM model"""
    if self.current_image is None:
        QMessageBox.warning(self, "Warning", "Please load an image first")
        return
    
    try:
        # get processing parameters
        downsample_ratio = self.downsample_ratio.value() if self.downsample_check.isChecked() else 1.0
        patch_size = self.patch_size.value() if self.patch_check.isChecked() else None
        
        # process image
        self.masks = self.interactive_helper.process_image(
            downsample_ratio=downsample_ratio,
            patch_size=patch_size
        )
        
        # update display
        self.current_mask_index = -1
        self.display_image(self.current_image_path)
        
        # display result
        self.result_label.setText(f"Detection completed! Found {len(self.masks)} regions")
        
    except Exception as e:
        self.result_label.setText(f"Detection failed: {str(e)}")
        logging.error(f"Error in run_auto_detection: {str(e)}")
        logging.error(traceback.format_exc())

def display_image(self, image_path):
    """Display image with masks"""
    try:
        # get visualization result
        vis_image = self.interactive_helper.get_visualization(self.current_mask_index)
        
        # convert to QImage
        height, width, channel = vis_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(vis_image.data, width, height, 
                       bytes_per_line, QImage.Format_RGB888)
        
        # display image
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
        
    except Exception as e:
        logging.error(f"Error in display_image: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def closeEvent(self, event):
    """Clean up resources when closing the application"""
    if self.interactive_helper is not None:
        self.interactive_helper.cleanup()
    event.accept()

# SAM model checkpoints
sam_checkpoints = {
    'vit_h': 'sam_vit_h_4b8939.pth',
    'vit_l': 'sam_vit_l_0b3195.pth',
    'vit_b': 'sam_vit_b_01ec64.pth'
}

if __name__ == '__main__':
    # register Section Counter callbacks
    register_section_counter_callbacks(app)
    # register Section Sequencing callbacks
    register_sequencing_callbacks(app)
    section_order_overlap.register_overlap_callbacks(app)
    # production mode: close Flask reloader, avoid multiple browser opens due to multiple processes
    import os as _os
    _port = int(_os.getenv('PORT', '8050'))
    app.run(host='127.0.0.1', port=_port, debug=False, use_reloader=False)