import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import io
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import ast
import tifffile
from modules.sequencing.sequential_alignment import draw_global_match_lines_with_overlap_split_aligned
from modules.sequencing.generate_aligned_tif_stack import generate_aligned_tif_stack

# ========== UI layout ==========
overlap_layout = html.Div([
    html.H2("Order Visualization", className="mb-4 text-primary"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Upload Images (multiple allowed):", className="fw-bold mb-2"),
                dcc.Upload(
                    id="overlap-image-upload",
                    children=dbc.Button("Upload Images", color="primary", className="w-100 mb-3"),
                    multiple=True,
                    accept="image/*",
                ),
                html.Label("Upload Mask CSVs (multiple allowed):", className="fw-bold mb-2 mt-3"),
                dcc.Upload(
                    id="overlap-csv-upload",
                    children=dbc.Button("Upload Mask CSVs", color="secondary", className="w-100 mb-3"),
                    multiple=True,
                    accept=".csv",
                ),
                html.Label("Upload Final Chain TXT:", className="fw-bold mb-2 mt-3"),
                dcc.Upload(
                    id="overlap-txt-upload",
                    children=dbc.Button("Upload Chain TXT", color="dark", className="w-100 mb-3"),
                    multiple=False,
                    accept=".txt",
                ),
                html.Div(id="overlap-image-list", className="mb-2"),
                html.Div(id="overlap-csv-list", className="mb-2"),
                html.Div(id="overlap-txt-list", className="mb-2"),
            ], className="p-4 bg-light rounded shadow-sm h-100"),
        ], width=3),
        dbc.Col([
            html.Div([
                dbc.Button("Generate Overlay", id="generate-overlay-btn", color="success", className="mb-3 w-100"),
                dcc.Loading(
                    id="overlay-loading",
                    type="default",
                    children=html.Div(id="overlay-output", className="d-flex justify-content-center align-items-center", style={"minHeight": "500px", "background": "#f5f7fa", "borderRadius": "12px"})
                ),
            ], className="p-4 bg-white rounded shadow-sm h-100"),
        ], width=9),
    ], className="g-4"),
    html.Hr(className="my-4"),
    html.H3("Stack & Alignment", className="mb-3 text-primary"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Original Images Folder Path:", className="fw-bold mb-2"),
                dcc.Input(id="stack-folder-input", type="text", placeholder="Enter folder path...", className="w-100", style={"minWidth": "180px"}),
                html.Label("Upload Chain TXT File:", className="fw-bold mb-2", style={"marginTop": "20px"}),
                dcc.Upload(
                    id="stack-txt-upload",
                    children=dbc.Button("Upload Chain TXT", color="dark", className="w-100", style={"minWidth": "180px"}),
                    multiple=False,
                    accept=".txt",
                ),
                html.Div(id="stack-txt-list", className="mb-2"),
            ], style={"padding": "20px", "background": "#f8f9fa", "borderRadius": "10px"})
        ], width=3),
        dbc.Col([
            html.Div([
                dbc.Button("Generate Stack Alignment", id="generate-stack-btn", color="success", className="w-100", style={"minWidth": "180px"}),
                dcc.Loading(
                    id="stack-loading",
                    type="default",
                    children=html.Div(id="stack-output", className="d-flex justify-content-center align-items-center", style={"minHeight": "500px", "background": "#f5f7fa", "borderRadius": "12px"})
                ),
                dbc.Button("Visualization", id="generate-tif-stack-btn", color="primary", className="w-100", style={"minWidth": "180px", "marginTop": "30px"}),
                dcc.Loading(
                    id="tif-stack-loading",
                    type="default",
                    children=html.Div(id="tif-stack-output", className="d-flex justify-content-center align-items-center", style={"minHeight": "500px", "background": "#f5f7fa", "borderRadius": "12px"})
                ),
            ], style={"padding": "20px", "background": "#fff", "borderRadius": "10px"})
        ], width=9)
    ], align="start", className="g-2"),
])

# ========== backend logic ==========
def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert('RGBA')
    return img

def parse_csv(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

def parse_chain_txt(contents):
    import re
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    # extract STEP 5 part
    m = re.search(r'STEP 5.*?final single chain.*?[:：]\s*(.*?)\n', decoded, re.DOTALL)
    if m:
        chain_line = m.group(1)
        # extract numbers of section_x
        ids = [s.split('_')[1] for s in chain_line.split('->') if 'section_' in s]
        return ids, chain_line
    return [], ""

def draw_overlay(images, csvs, order_map=None):
    # use the first image as the base image
    if not images or not csvs:
        return None
    base_img = images[0].copy()
    draw = ImageDraw.Draw(base_img)


    for df in csvs:
        for _, row in df.iterrows():
            # ---- 1) parse center coordinates --------------------------------------
            raw = row.get("center_coordinates")
            if raw is None:
                continue

            # convert string to Python object safely
            if isinstance(raw, str):
                coords = ast.literal_eval(raw)
            else:
                coords = raw

            # maybe [x, y] or [[x, y]]
            if isinstance(coords[0], (list, tuple)):
                x, y = coords[0][:2]
            else:
                x, y = coords[:2]
            x, y = int(x), int(y)

            # ---- 2) parse section id -----------------------------------
            name = str(row.get("id", "?"))
            section_id = name.split("_", 1)[1] if name.startswith("section_") else name

            # ---- 3) decide tag content ---------------------------------------
            if order_map and section_id in order_map:
                tag = order_map[section_id]
            else:
                tag = section_id

            # ---- 4) draw background rectangle + text ---------------------------------

            # ➊ calculate font size / pad
            W, H = base_img.size
            font_size = max(int(min(W, H) * 0.02), 36)
            pad = int(font_size * 0.4)

            # ➋ try to load common fonts
            font = None
            for path in [
                "arial.ttf",                                   # Windows default
                "/Library/Fonts/Arial.ttf",                    # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux + Pillow
            ]:
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except IOError:
                    continue

            if font is None:
                raise RuntimeError(
                    "TrueType font not found. "
                    "Install Arial or DejaVuSans, or update the font path in draw_overlay()."
                )

            pad = int(font_size * 0.4)

            # ➋ use font to calculate text width and height
            bbox = font.getbbox(tag)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # ➌ draw rectangle & text
            rect = [x - w//2 - pad, y - h//2 - pad,
                    x + w//2 + pad, y + h//2 + pad]

            draw.rectangle(rect, fill=(0, 0, 0, 200))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:   # black outline
                draw.text((x - w//2 + dx, y - h//2 + dy),
                        tag, font=font, fill=(0, 0, 0, 255))
            draw.text((x - w//2, y - h//2),
                    tag, font=font, fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    base_img.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{data}"

def ensure_vis_folder():
    # unified visualization dir under results/order_viz/<timestamp>
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    base = os.path.join(root, 'results', 'order_viz')
    os.makedirs(base, exist_ok=True)
    ts_dir = os.path.join(base, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(ts_dir, exist_ok=True)
    return ts_dir

def get_chain_txt_from_upload(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    vis_dir = ensure_vis_folder()
    chain_path = os.path.join(vis_dir, 'uploaded_chain.txt')
    with open(chain_path, 'w') as f:
        f.write(decoded)
    return chain_path

# register Dash callbacks

def register_overlap_callbacks(app):
    @app.callback(
        Output("overlap-image-list", "children"),
        Input("overlap-image-upload", "filename"),
        prevent_initial_call=True
    )
    def show_image_list(filenames):
        if not filenames:
            return "No images uploaded."
        return html.Ul([html.Li(f) for f in filenames])

    @app.callback(
        Output("overlap-csv-list", "children"),
        Input("overlap-csv-upload", "filename"),
        prevent_initial_call=True
    )
    def show_csv_list(filenames):
        if not filenames:
            return "No CSVs uploaded."
        return html.Ul([html.Li(f) for f in filenames])

    @app.callback(
        Output("overlap-txt-list", "children"),
        Input("overlap-txt-upload", "filename"),
        prevent_initial_call=True
    )
    def show_txt_list(filenames):
        if not filenames:
            return "No chain TXT uploaded."
        return html.Ul([html.Li(f) for f in [filenames] if f])

    @app.callback(
        Output("overlay-output", "children"),
        Input("generate-overlay-btn", "n_clicks"),
        State("overlap-image-upload", "contents"),
        State("overlap-csv-upload", "contents"),
        State("overlap-txt-upload", "contents"),
        prevent_initial_call=True
    )
    def generate_overlay(n, image_contents, csv_contents, txt_contents):
        if not n or not image_contents or not csv_contents:
            return "Please upload images and CSVs."
        images = [parse_image(c) for c in image_contents]
        csvs = [parse_csv(c) for c in csv_contents]
        order_map = None
        if txt_contents:
            ids, _ = parse_chain_txt(txt_contents)
            # ids is the list of section ids (e.g. ['6','1',...])
            # order_map: section_id -> order number
            order_map = {sid: str(i+1) for i, sid in enumerate(ids)}
        # generate overlay image
        base_img = images[0].copy()
        draw = ImageDraw.Draw(base_img)
        for df in csvs:
            for _, row in df.iterrows():
                raw = row.get("center_coordinates")
                if raw is None:
                    continue
                if isinstance(raw, str):
                    coords = ast.literal_eval(raw)
                else:
                    coords = raw
                if isinstance(coords[0], (list, tuple)):
                    x, y = coords[0][:2]
                else:
                    x, y = coords[:2]
                x, y = int(x), int(y)
                name = str(row.get("id", "?"))
                section_id = name.split("_", 1)[1] if name.startswith("section_") else name
                if order_map and section_id in order_map:
                    tag = order_map[section_id]
                else:
                    tag = section_id
                W, H = base_img.size
                font_size = max(int(min(W, H) * 0.02), 36)
                pad = int(font_size * 0.4)
                font = None
                for path in [
                    "arial.ttf",
                    "/Library/Fonts/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                ]:
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except IOError:
                        continue
                if font is None:
                    raise RuntimeError(
                        "TrueType font not found. "
                        "Install Arial or DejaVuSans, or update the font path in draw_overlay()."
                    )
                pad = int(font_size * 0.4)
                bbox = font.getbbox(tag)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                rect = [x - w//2 - pad, y - h//2 - pad,
                        x + w//2 + pad, y + h//2 + pad]
                draw.rectangle(rect, fill=(0, 0, 0, 200))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    draw.text((x - w//2 + dx, y - h//2 + dy),
                            tag, font=font, fill=(0, 0, 0, 255))
                draw.text((x - w//2, y - h//2),
                        tag, font=font, fill=(255, 255, 255, 255))
        # save to unified directory
        vis_dir = ensure_vis_folder()
        overlay_path = os.path.join(vis_dir, 'overlay_result.png')
        base_img.save(overlay_path)
        try:
            from modules.common.io import save_meta
            save_meta(vis_dir, {
                'module': 'order_viz',
                'artifact': 'overlay_result.png'
            })
        except Exception:
            pass
        # convert to base64 and return
        buf = io.BytesIO()
        base_img.save(buf, format='PNG')
        data = base64.b64encode(buf.getvalue()).decode()
        return html.Img(src=f"data:image/png;base64,{data}", style={"maxWidth": "100%", "border": "2px solid #888"})

    @app.callback(
        Output("stack-txt-list", "children"),
        Input("stack-txt-upload", "filename"),
        prevent_initial_call=True
    )
    def show_stack_txt_filename(filename):
        if not filename:
            return "No chain TXT uploaded."
        return html.Div(f"Uploaded: {filename}", style={"color": "#007bff", "fontWeight": "bold"})

    @app.callback(
        Output("stack-output", "children"),
        Input("generate-stack-btn", "n_clicks"),
        State("stack-folder-input", "value"),
        State("stack-txt-upload", "contents"),
        prevent_initial_call=True
    )
    def generate_stack_alignment(n, folder, chain_txt_contents):
        if not n or not folder or not chain_txt_contents:
            return "Please provide folder path and upload chain TXT."
        vis_dir = ensure_vis_folder()
        chain_path = get_chain_txt_from_upload(chain_txt_contents)
        output_path = os.path.join(vis_dir, 'global_match_lines_with_overlap_split_aligned_thumb.png')
        # only generate one thumbnail, resize=0.1
        draw_global_match_lines_with_overlap_split_aligned(folder, chain_path, resize=0.1, output_path=output_path)
        with open(output_path, 'rb') as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        return html.Img(src=f"data:image/png;base64,{img_b64}", style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"})

    @app.callback(
        Output("tif-stack-output", "children"),
        Input("generate-tif-stack-btn", "n_clicks"),
        State("stack-folder-input", "value"),
        State("stack-txt-upload", "contents"),
        prevent_initial_call=True
    )
    def generate_tif_stack(n, folder, chain_txt_contents):
        if not n or not folder or not chain_txt_contents:
            return "Please provide folder path and upload chain TXT."
        vis_dir = ensure_vis_folder()
        chain_path = get_chain_txt_from_upload(chain_txt_contents)
        output_path = os.path.join(vis_dir, 'aligned_stack_thumb.tif')
        # only generate one thumbnail, resize=0.1
        generate_aligned_tif_stack(folder, chain_path, resize=0.1, output_path=output_path)
        # append to unified results directory (does not affect original output)
        try:
            from modules.common.paths import get_run_dir
            from modules.common.io import copy_to, save_meta
            run_dir = get_run_dir('order_viz')
            copy_to(run_dir, [(output_path, 'aligned_stack_thumb.tif')])
            save_meta(run_dir, {
                'module': 'order_viz',
                'artifact': 'aligned_stack_thumb.tif'
            })
        except Exception:
            pass
        stack = tifffile.imread(output_path)
        n_slices = stack.shape[0]
        def get_img_b64(idx):
            img = stack[idx]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            _, buf = cv2.imencode('.png', img)
            return base64.b64encode(buf.tobytes()).decode()
        img_b64 = get_img_b64(0)
        return html.Div([
            html.Div([
                html.Button('Prev', id='tif-stack-prev', n_clicks=0, style={
                    'marginRight': '20px',
                    'fontSize': '22px',
                    'padding': '10px 28px',
                    'border': '2px solid #222',
                    'borderRadius': '8px',
                    'background': '#f8f9fa',
                    'minWidth': '90px',
                }),
                html.Div(
                    id='tif-stack-index',
                    children='0',
                    style={
                        'fontSize': '22px',
                        'padding': '8px 24px',
                        'background': '#666',
                        'color': 'white',
                        'borderRadius': '10px',
                        'display': 'inline-block',
                        'margin': '0 24px',
                        'minWidth': '48px',
                        'textAlign': 'center',
                    }
                ),
                html.Button('Next', id='tif-stack-next', n_clicks=0, style={
                    'marginLeft': '20px',
                    'fontSize': '22px',
                    'padding': '10px 28px',
                    'border': '2px solid #222',
                    'borderRadius': '8px',
                    'background': '#f8f9fa',
                    'minWidth': '90px',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '28px'}),
            dcc.Slider(
                id='tif-stack-slider',
                min=0,
                max=n_slices-1,
                step=1,
                value=0,
                marks={i:str(i+1) for i in range(n_slices)},
                updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": True},
                className='tif-stack-slider'
            ),
            html.Img(id='tif-stack-img', src=f'data:image/png;base64,{img_b64}', style={"maxWidth": "100%", "height": "auto", "display": "block", "margin": "auto"}),
            dcc.Store(id='tif-stack-store', data={'stack_path': output_path, 'n_slices': n_slices})
        ])

    @app.callback(
        Output('tif-stack-img', 'src'),
        Output('tif-stack-index', 'children'),
        Input('tif-stack-slider', 'value'),
        State('tif-stack-store', 'data'),
        prevent_initial_call=True
    )
    def update_tif_stack_img(idx, store):
        if not store or 'stack_path' not in store:
            raise dash.exceptions.PreventUpdate
        stack = tifffile.imread(store['stack_path'])
        img = stack[idx]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode('.png', img)
        img_b64 = base64.b64encode(buf.tobytes()).decode()
        return f'data:image/png;base64,{img_b64}', str(idx)

    @app.callback(
        Output('tif-stack-slider', 'value'),
        [Input('tif-stack-prev', 'n_clicks'), Input('tif-stack-next', 'n_clicks')],
        State('tif-stack-slider', 'value'),
        State('tif-stack-store', 'data'),
        prevent_initial_call=True
    )
    def change_tif_stack_slice(prev, next_, value, store):
        if not store or 'n_slices' not in store:
            raise dash.exceptions.PreventUpdate
        triggered = dash.callback_context.triggered_id
        n_slices = store['n_slices']
        if triggered == 'tif-stack-prev' and value > 0:
            return value - 1
        elif triggered == 'tif-stack-next' and value < n_slices - 1:
            return value + 1
        return value 
