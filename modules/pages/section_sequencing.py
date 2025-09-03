import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import os
import subprocess
import threading
import time

# ========== UI layout ==========
sequencing_layout = html.Div([
    html.H2("Section Sequencing", className="mb-4 text-primary"),
    # 1. Select Images Folder (multi-file)
    html.Div([
        html.Label("Images Folder Path:"),
        dcc.Input(
            id="folder-input", 
            type="text", 
            placeholder="Enter folder path or drag folder here...", 
            style={"width": "100%", "marginBottom": "10px"}
        ),
        dbc.Button("Scan Images", id="scan-images-btn", color="primary"),
        html.Span(id="selected-folder-path", style={"marginLeft": "10px"}),
        dcc.Store(id="images-folder-store"),
    ], className="mb-3"),

    # 2. SIFT Parameter Adjustment (4 columns)
    dbc.Row([
        dbc.Col([
            html.Label(id='sift-resize-label', children='Resize (0.15)'),
            dcc.Slider(id='sift-resize-slider', min=0.1, max=1.0, step=0.01, value=0.15, marks={0.1:'0.1',0.5:'0.5',1.0:'1.0'}),
            html.Label(id='cpu-workers-label', children='CPU Workers (6)'),
            dcc.Slider(id='cpu-workers-slider', min=1, max=16, step=1, value=6, marks={1:'1',4:'4',8:'8',12:'12',16:'16'}),
        ], width=3),
        dbc.Col([
            html.Label(id='sift-features-label', children='SIFT Features (3000)'),
            dcc.Slider(id='sift-features-slider', min=1000, max=5000, step=500, value=3000, marks={1000:'1000',3000:'3000',5000:'5000'}),
            html.Label(id='sift-contrast-label', children='SIFT Contrast (0.02)'),
            dcc.Slider(id='sift-contrast-slider', min=0.01, max=0.1, step=0.01, value=0.02, marks={0.01:'0.01',0.05:'0.05',0.1:'0.1'}),
            html.Label(id='sift-edge-label', children='SIFT Edge (20)'),
            dcc.Slider(id='sift-edge-slider', min=10, max=50, step=1, value=20, marks={10:'10',20:'20',50:'50'}),
        ], width=3),
        dbc.Col([
            html.Label(id='flann-trees-label', children='FLANN Trees (8)'),
            dcc.Slider(id='flann-trees-slider', min=4, max=16, step=1, value=8, marks={4:'4',8:'8',16:'16'}),
            html.Label(id='flann-checks-label', children='FLANN Checks (100)'),
            dcc.Slider(id='flann-checks-slider', min=50, max=200, step=10, value=100, marks={50:'50',100:'100',200:'200'}),
        ], width=3),
        dbc.Col([
            html.Label(id='lowe-ratio-label', children="Lowe's Ratio (0.85)"),
            dcc.Slider(id='lowe-ratio-slider', min=0.7, max=0.9, step=0.01, value=0.85, marks={0.7:'0.7',0.8:'0.8',0.9:'0.9'}),
            html.Label(id='ransac-threshold-label', children='RANSAC Threshold (25)'),
            dcc.Slider(id='ransac-threshold-slider', min=10, max=50, step=1, value=25, marks={10:'10',25:'25',50:'50'}),
            html.Label(id='min-inlier-ratio-label', children='Min Inlier Ratio (0.08)'),
            dcc.Slider(id='min-inlier-ratio-slider', min=0.05, max=0.15, step=0.01, value=0.08, marks={0.05:'0.05',0.1:'0.1',0.15:'0.15'}),
        ], width=3),
    ], style={"marginBottom": "20px"}),

    # 3. SIFT Run and Log
    dbc.Row([
        dbc.Col([
            dbc.Button('Run SIFT Pairwise Alignment', id='run-sift-alignment-btn', color='primary', className='w-100 mb-3', disabled=True),
        ], width=6),
        dbc.Col([
            dbc.Button('Stop SIFT', id='stop-sift-btn', color='danger', className='w-100 mb-3', disabled=True),
        ], width=6),
    ]),
    html.Div([
        dcc.Interval(id="sift-log-interval", interval=300, n_intervals=0, disabled=True),
        html.Pre(id="sift-log-output", style={"height": "300px", "overflowY": "auto", "background": "#222", "color": "#eee"})
    ]),

    # 4A. Clean CSV (upload → clean)
    html.Hr(),
    html.H5("Step A · Clean raw pair-wise CSV"),

    dbc.Row([                                                    # <<<
        dbc.Col(                                                # <<<
            dcc.Upload(                                         # upload button wrapped in Upload
                id="raw-csv-upload",
                children=dbc.Button("Upload raw CSV",
                                    color="secondary",
                                    className="w-100"),         # full width
                multiple=False,
                accept=".csv",
                style={"width": "100%"},
            ),
            width=6                                             # half width
        ),
        dbc.Col(
            dbc.Button("Clean CSV Results",
                    id="clean-csv-btn",
                    color="success",
                    className="w-100",
                    disabled=True),
            width=6
        ),
    ]),                                                         # <<<

    html.Span(id="raw-csv-filename", className="ms-2 fst-italic"),
    dcc.Store(id="csv-file-store"),

    # 4B. Build Chains (upload → build)
    html.Hr(),
    html.H5("Step B · Build section chains from CLEAN csv"),

    dbc.Row([                                                    # <<<
        dbc.Col(
            dcc.Upload(
                id="clean-csv-upload",
                children=dbc.Button("Upload CLEAN csv",
                                    color="secondary",
                                    className="w-100"),
                multiple=False,
                accept=".csv",
                style={"width": "100%"},
            ),
            width=6
        ),
        dbc.Col(
            dbc.Button("Build Section Chains",
                    id="build-chains-btn",
                    color="info",
                    className="w-100",
                    disabled=True),
            width=6
        ),
    ]),                                                         # <<<

    html.Span(id="clean-csv-filename", className="ms-2 fst-italic"),
    dcc.Store(id="clean-csv-file-store"),    

    # Results
    # html.Div(id='sequencing-status', className='mt-3'),
    html.Div(id='sequencing-results-display', className='border p-3', style={'minHeight': '300px'}),
    dcc.Store(id="sift-log-path"),
])

# ========== backend log thread ==========
sift_log_lines = []
sift_log_lock = threading.Lock()
sift_running = False

def run_sift_and_log(cmd, log_path):
    global sift_running
    print(f"[DEBUG] SIFT thread started, log_path={log_path}")
    sift_running = True
    try:
        with open(log_path, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                f.write(line)
                f.flush()
                with sift_log_lock:
                    sift_log_lines.append(line)
                print(f"[DEBUG] SIFT log: {line.strip()}")
            proc.wait()
    except Exception as e:
        print(f"[ERROR] SIFT thread exception: {e}")
    sift_running = False
    print(f"[DEBUG] SIFT thread ended")

# ========== callback registration ==========
def register_sequencing_callbacks(app):
    # 1. Select Images Folder, take common prefix as folder path
    @app.callback(
        Output("images-folder-store", "data"),
        Output("selected-folder-path", "children"),
        Input("folder-input", "value"),
        Input("scan-images-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def scan_folder(folder_path, scan_clicks):
        if not folder_path:
            raise dash.exceptions.PreventUpdate
        
        if scan_clicks is None: # Only scan if button was clicked
            return dash.no_update, "No folder path entered."

        if not os.path.exists(folder_path):
            return dash.no_update, f"Folder not found: {folder_path}"
        
        tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        
        if not tif_files and not png_files:
            return dash.no_update, f"No .tif or .png files found in {folder_path}"
        
        return folder_path, f"Found {len(tif_files)} .tif files and {len(png_files)} .png files in {folder_path}"

    # 2. After selecting folder, Run SIFT button is available
    @app.callback(
        Output('run-sift-alignment-btn', 'disabled'),
        Input('images-folder-store', 'data'),
        prevent_initial_call=True
    )
    def enable_run_sift(folder):
        return not bool(folder)

    # 2.5. Enable/disable Stop button based on SIFT running status
    @app.callback(
        Output('stop-sift-btn', 'disabled'),
        Input('sift-log-interval', 'disabled'),
        prevent_initial_call=True
    )
    def enable_stop_sift(interval_disabled):
        return interval_disabled

    # 3. Run SIFT button starts SIFT and outputs in real-time (only writes to sequencing-status)
    @app.callback(
        # Output('sequencing-status', 'children'),
        Output('sift-log-path', 'data'),
        Output('sift-log-interval', 'disabled', allow_duplicate=True),      # <<< new
        Input('run-sift-alignment-btn', 'n_clicks'),
        State('images-folder-store', 'data'),
        State('sift-resize-slider', 'value'),
        State('cpu-workers-slider', 'value'),
        State('sift-features-slider', 'value'),
        State('sift-contrast-slider', 'value'),
        State('sift-edge-slider', 'value'),
        State('flann-trees-slider', 'value'),
        State('flann-checks-slider', 'value'),
        State('lowe-ratio-slider', 'value'),
        State('ransac-threshold-slider', 'value'),
        State('min-inlier-ratio-slider', 'value'),
        prevent_initial_call=True
    )
    def start_sift(n_clicks, folder, resize, cpu_workers, sift_features, sift_contrast, sift_edge, flann_trees, flann_checks, lowe_ratio, ransac_threshold, min_inlier_ratio):
        if not n_clicks or not folder:
            raise dash.exceptions.PreventUpdate
        import os
        # unified results root: <project>/results/sequencing
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        result_root = os.path.join(project_root, 'results', 'sequencing')
        sift_results_dir = os.path.join(result_root, 'sift_results')
        os.makedirs(sift_results_dir, exist_ok=True)
        out_dir = os.path.join(sift_results_dir, f"sift_pairwise_out_{int(time.time())}")
        # Resolve absolute path to modules/sequencing/sift_pairwise_alignment.py from project root
        sift_script = os.path.join(project_root, 'modules', 'sequencing', 'sift_pairwise_alignment.py')
        cmd = [
            'python3', sift_script,
            '--folder', folder,
            '--out_dir', out_dir,
            '--resize', str(resize),
            '--cpu_workers', str(cpu_workers),
            '--sift_features', str(sift_features),
            '--sift_contrast', str(sift_contrast),
            '--sift_edge', str(sift_edge),
            '--flann_trees', str(flann_trees),
            '--flann_checks', str(flann_checks),
            '--lowe_ratio', str(lowe_ratio),
            '--ransac_threshold', str(ransac_threshold),
            '--min_inlier_ratio', str(min_inlier_ratio),
            '--all_pairs'
        ]
        # place logs under results/sequencing/logs
        log_dir = os.path.join(result_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"sift_log_{int(time.time())}.txt")
        global sift_log_lines
        sift_log_lines = []
        threading.Thread(target=run_sift_and_log, args=(cmd, log_path), daemon=True).start()
        return (
            log_path,
            False                                           # <<< False = start polling
        )
    # 4. Interval to pull logs (only callback writing sift-log-output and sift-log-interval.disabled)
    # ---- keep Output order consistent ----
    @app.callback(
        Output('sift-log-output', 'children'),
        Output('sift-log-interval', 'disabled'),
        Input('sift-log-interval', 'n_intervals'),
        State('sift-log-path', 'data'),
        prevent_initial_call=True
    )
    def update_sift_log(_, log_path):
        log = ''
        if log_path and os.path.exists(log_path):
            with open(log_path, 'r', errors='ignore') as f:
                log = f.read()[-10000:]

        # determine if finished based on keywords (modify according to actual output in script)
        finished = ('thread ended' in log) or ('CSV written' in log)
        return log, finished            # <<< second return value controls disabled

    # 5. Select CSV file




    # 7. Clean CSV button handling
    @app.callback(
        Output('sequencing-results-display', 'children'),
        # Output('sequencing-status', 'children', allow_duplicate=True),
        Input('clean-csv-btn', 'n_clicks'),
        State('csv-file-store', 'data'),
        prevent_initial_call=True
    )
    def clean_csv(n, csv_path):
        if not n or not csv_path:
            raise dash.exceptions.PreventUpdate
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        # clean script: prefer modules/sequencing/clean_new_csv.py, fallback to archived path if needed
        candidate_scripts = [
            os.path.join(project_root, 'modules', 'sequencing', 'clean_new_csv.py'),
            os.path.join(project_root, 'clean_new_csv.py'),
            os.path.join(project_root, 'modules', 'legacy', 'clean_new_csv.py'),
        ]
        clean_script = next((p for p in candidate_scripts if os.path.exists(p)), candidate_scripts[0])
        # unified output path under results/sequencing/cleaned_csv
        cleaned_dir = os.path.join(project_root, 'results', 'sequencing', 'cleaned_csv')
        os.makedirs(cleaned_dir, exist_ok=True)
        base = os.path.basename(csv_path).rsplit('.', 1)[0]
        output_csv = os.path.join(cleaned_dir, f"{base}_cleaned.csv")
        cmd = ['python3', clean_script, csv_path, '-o', output_csv]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return html.Pre(result.stdout if result.returncode==0 else result.stderr)

        else:
            return html.Pre(result.stderr), "CSV cleaning failed"

    # 8. Select Cleaned CSV


    # 9. After selecting Cleaned CSV, Build button is available
 
    # 10. Build button handling
    @app.callback(
        Output('sequencing-results-display', 'children', allow_duplicate=True),
        # Output('sequencing-status', 'children', allow_duplicate=True),
        Input('build-chains-btn', 'n_clicks'),
        State('clean-csv-file-store', 'data'),
        prevent_initial_call=True
    )
    def build_chains(n, csv_path):
        if not n or not csv_path:
            raise dash.exceptions.PreventUpdate
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        result_root = os.path.join(project_root, 'results', 'sequencing', 'final_order_chain')
        os.makedirs(result_root, exist_ok=True)
        ts_dir = os.path.join(result_root, time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(ts_dir, exist_ok=True)
        output_file = os.path.join(ts_dir, 'chain_result.txt')
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        # prefer modules/sequencing version; fallback to archived root script
        candidates = [
            os.path.join(project_root, 'modules', 'sequencing', 'best_pair_chain_graph.py'),
            os.path.join(project_root, 'best_pair_chain_graph.py'),
            os.path.join(project_root, 'archive_unused_20250829', 'best_pair_chain_graph.py'),
        ]
        chains_script = next((p for p in candidates if os.path.exists(p)), candidates[0])
        cmd = ['python3', chains_script, '--csv', csv_path, '--output', output_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            info_lines = result.stdout.strip().splitlines()
            info = '\n'.join(info_lines[:2])  # only take first two lines
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                txt_content = f.read()
            display_text = info + '\n' + txt_content if info else txt_content
            # write meta to current time folder
            try:
                from modules.common.io import save_meta
                save_meta(ts_dir, {
                    'module': 'sequencing',
                    'artifact': 'chain_result.txt'
                })
            except Exception:
                pass
            return html.Pre(display_text)
        else:
            return html.Pre(result.stderr), "Section chains building failed"

    # 11. Stop SIFT button handling
    @app.callback(
        # Output('sequencing-status', 'children', allow_duplicate=True),
        Output('sift-log-interval', 'disabled', allow_duplicate=True),   # <<< 新增
        Input('stop-sift-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def stop_sift(n):
        if not n:
            raise dash.exceptions.PreventUpdate
        try:
            subprocess.run(['pkill', '-f', 'sift_pairwise_alignment.py'], check=False)
            return True      # <<< True = stop polling immediately
        except Exception as e:
            return f"Failed to stop SIFT: {e}", True
    # --- A. upload raw CSV → enable Clean button -------------------------------
    @app.callback(
        Output('csv-file-store', 'data'),
        Output('raw-csv-filename', 'children'),
        Output('clean-csv-btn', 'disabled'),
        Input('raw-csv-upload', 'contents'),
        State('raw-csv-upload', 'filename'),
        prevent_initial_call=True
    )
    def store_raw_csv(contents, fname):
        if contents is None:
            raise dash.exceptions.PreventUpdate
        import base64, tempfile, uuid, os
        _, encoded = contents.split(',', 1)
        data = base64.b64decode(encoded)
        path = os.path.join(tempfile.gettempdir(),
                            f"raw_{uuid.uuid4().hex}_{fname}")
        with open(path, 'wb') as f:
            f.write(data)
        return path, html.Code(fname), False   # disabled=False

    # --- B. upload CLEAN csv → enable Build button -----------------------------
    @app.callback(
        Output('clean-csv-file-store', 'data'),
        Output('clean-csv-filename', 'children'),
        Output('build-chains-btn', 'disabled'),
        Input('clean-csv-upload', 'contents'),
        State('clean-csv-upload', 'filename'),
        prevent_initial_call=True
    )
    def store_clean_csv(contents, fname):
        if contents is None:
            raise dash.exceptions.PreventUpdate
        import base64, tempfile, uuid, os
        _, encoded = contents.split(',', 1)
        data = base64.b64decode(encoded)
        path = os.path.join(tempfile.gettempdir(),
                            f"clean_{uuid.uuid4().hex}_{fname}")
        with open(path, 'wb') as f:
            f.write(data)
        return path, html.Code(fname), False   # disabled=False
 
    # parameter label shows current value in real-time
    @app.callback(Output('sift-resize-label', 'children'), Input('sift-resize-slider', 'value'))
    def update_resize_label(v):
        return f'Resize ({v:.2f})'
    @app.callback(Output('cpu-workers-label', 'children'), Input('cpu-workers-slider', 'value'))
    def update_cpu_workers_label(v):
        return f'CPU Workers ({v})'
    @app.callback(Output('sift-features-label', 'children'), Input('sift-features-slider', 'value'))
    def update_sift_features_label(v):
        return f'SIFT Features ({v})'
    @app.callback(Output('sift-contrast-label', 'children'), Input('sift-contrast-slider', 'value'))
    def update_sift_contrast_label(v):
        return f'SIFT Contrast ({v:.2f})'
    @app.callback(Output('sift-edge-label', 'children'), Input('sift-edge-slider', 'value'))
    def update_sift_edge_label(v):
        return f'SIFT Edge ({v})'
    @app.callback(Output('flann-trees-label', 'children'), Input('flann-trees-slider', 'value'))
    def update_flann_trees_label(v):
        return f'FLANN Trees ({v})'
    @app.callback(Output('flann-checks-label', 'children'), Input('flann-checks-slider', 'value'))
    def update_flann_checks_label(v):
        return f'FLANN Checks ({v})'
    @app.callback(Output('lowe-ratio-label', 'children'), Input('lowe-ratio-slider', 'value'))
    def update_lowe_ratio_label(v):
        return f"Lowe's Ratio ({v:.2f})"
    @app.callback(Output('ransac-threshold-label', 'children'), Input('ransac-threshold-slider', 'value'))
    def update_ransac_threshold_label(v):
        return f'RANSAC Threshold ({v})'
    @app.callback(Output('min-inlier-ratio-label', 'children'), Input('min-inlier-ratio-slider', 'value'))
    def update_min_inlier_ratio_label(v):
        return f'Min Inlier Ratio ({v:.2f})'
 
