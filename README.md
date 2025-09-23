# WaferTools - Software for High-Throughput Integrative Connectomics Project

Copyright © 2025 Harvard University Lichtman Lab.

Part of the High-Throughput Integrative Mouse Connectomics (HI-MC) project — an international collaboration among Harvard, Princeton, MIT, Cambridge, Google, the Allen Institute, and Johns Hopkins.

This software provides an end-to-end workflow for wafer-based connectomics preprocessing:

- Section segmentation with SAM (Segment Anything)
- Section sequencing via SIFT pairwise alignment, cleaning, and chain building
- Order visualization and aligned stack preview
- Unified, reproducible result storage

While developed for HI-MC, the modules are broadly applicable to wafer-based connectomics and similar serial-section pipelines.

---

## Table of Contents

- [Quick start](#quick-start)
- [Online install](#online-install)
- [Offline install (with prebuilt wheels)](#offline-install-with-prebuilt-wheels)
- [Run](#run)
- [Modules and workflows](#modules-and-workflows)
  - [Module index (at a glance)](#module-index-at-a-glance)
  - [1) Section Counter (segmentation)](#1-section-counter-segmentation)
  - [2) Sequencing (pairwise alignment--clean--chain)](#2-sequencing-pairwise-alignment--clean--chain)
  - [3) Order Visualization](#3-order-visualization)
- [Project structure (selected)](#project-structure-selected)
- [Dependencies](#dependencies)
- [Common issues](#common-issues)
- [Citation and licensing](#citation-and-licensing)
- [Contact](#contact)

---

## Quick start

- Requirements: Python 3.10–3.12, macOS/Windows/Linux.
- GPU optional. SAM will use CUDA if available, otherwise CPU.
- Place SAM checkpoints in the project root (recommended) or `cache/`:
  *(If you don't have them, the software will download on first segmentation run — may take a few minutes.)*  
  `sam_vit_h.pth`, `sam_vit_l.pth`, `sam_vit_b.pth`

---

## Online install

```bash
cd Polygon_tool_V3
python3 -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
# Use your own requirements (or our offline set below)
pip install -r offline/requirements_offline.txt
```

## Offline install (with prebuilt wheels)
```bash
# Place wheels under offline/wheels/ for your OS/arch
pip install --no-index --find-links=offline/wheels -r offline/requirements_offline.txt
```

## Run
```bash
python app.py
# open http://127.0.0.1:8050
```

---

## Modules and workflows

### Module index (at a glance)

| Module | Goal | Key Outputs (under `results/`) |
|---|---|---|
| **Section Counter** | Detect wafer sections using SAM; optional expected-count filtering | `section_counter/<ts>/mask.png`, `sections.csv`, `masks.pkl`, `meta.json` |
| **Sequencing** | Build plausible section order (SIFT pairwise → clean → chain) | `sequencing/sift_results/.../pairwise_alignment_results.csv`, `cleaned_csv/<input>_cleaned.csv`, `final_order_chain/<ts>/chain_result.txt`, `meta.json` |
| **Order Visualization** | Visual check of order, matches, and aligned stack | `order_viz/<ts>/overlay_result.png`, `global_match_lines_with_overlap_split_aligned_thumb.png`, `aligned_stack_thumb.tif`, `meta.json` |

### 1) Section Counter (segmentation)

- **Goal:** Detect wafer sections using SAM; optionally filter by *Expected sections* with area clustering instead of absolute thresholds.
- **Steps:**
  1. Upload wafer image (PNG/TIF).
  2. Select SAM model (vit_b/l/h). Checkpoints are auto-located in project root or `cache/`.
  3. Optionally set *Expected sections*. Click **Auto Detect**.
  4. Optional **Filtering** refines by expected count or controlled area range.
  5. **Export** writes results.
- **Outputs (unified):**
  - `results/section_counter/<timestamp>/mask.png`
  - `results/section_counter/<timestamp>/sections.csv`
  - `results/section_counter/<timestamp>/masks.pkl`
  - `results/section_counter/<timestamp>/meta.json`
- **Notes:**
  - Debug intermediate PNGs are disabled by default and will not be generated.
  - Legacy folder `Result_masking` is disabled by default. To enable for backward compatibility:
    - macOS/Linux: `export WAFER_ENABLE_LEGACY_RESULTS=1`
    - Windows: `set WAFER_ENABLE_LEGACY_RESULTS=1`

**Advanced options (env vars):**
```bash
WAFER_ENABLE_LEGACY_RESULTS=0|1  # default 0
WAFER_SAVE_DEBUG_VIS=0|1         # default 0
```

---

### 2) Sequencing (pairwise alignment → clean → chain)

- **Goal:** Build plausible section order from a folder of section images.
- **Steps:**
  1. Enter/scan the images folder.
  2. **Run SIFT Pairwise Alignment** generates pairwise matches and writes a CSV.
  3. Upload the pairwise CSV to Step A and click **Clean CSV Results** to filter SSIM/scale and add a score.
  4. Upload the cleaned CSV to Step B and click **Build Section Chains**.
- **Outputs (unified):**
  - Pairwise CSV: `results/sequencing/sift_results/sift_pairwise_out_<ts>/pairwise_alignment_results.csv`
  - SIFT logs (for UI): `results/sequencing/logs/sift_log_<ts>.txt`
  - Cleaned CSV: `results/sequencing/cleaned_csv/<input>_cleaned.csv`
  - Chain result: `results/sequencing/final_order_chain/<timestamp>/chain_result.txt`
  - Plus `meta.json` alongside major artifacts

**Implementation notes:**
- Uses SIFT + FLANN + RANSAC; parameters adjustable in the UI.
- Cleaning keeps scale within 0.9–1.1, removes `ssim == -1`, computes `score = ssim × num_inliers`.

---

### 3) Order Visualization

- **Goal:** Visual check of order, masks, and aligned stack.
- **Steps:**
  1. Upload the reference image(s) and the mask CSV(s) (from Section Counter).
  2. Optionally upload the final chain TXT (from Sequencing).
  3. Click **Visualization** or **Build Aligned Stack**.
- **Outputs (unified):**
  - `results/order_viz/<timestamp>/overlay_result.png`
  - `results/order_viz/<timestamp>/global_match_lines_with_overlap_split_aligned_thumb.png`
  - `results/order_viz/<timestamp>/aligned_stack_thumb.tif`
  - `results/order_viz/<timestamp>/meta.json`

---

## Project structure (selected)

```text
modules/
  pages/
    section_counter.py         # Section segmentation UI
    section_sequencing.py      # SIFT, clean, chain UI
    section_order_overlap.py   # Visualization UI
  sequencing/
    sift_pairwise_alignment.py # SIFT pairwise CLI
    clean_new_csv.py           # Clean CSV CLI
    best_pair_chain_graph.py   # Chain builder CLI (adapter)
    generate_aligned_tif_stack.py
  section_counter/
    downsampled_sam.py         # SAM wrapper with downsampling
  modules/common/
    io.py, paths.py            # unified result I/O
results/
  section_counter/ …          # per-run folders with meta.json
  sequencing/ …               # logs, cleaned_csv, final_order_chain
  order_viz/ …                # visualization outputs
app.py                        # Dash entrypoint
offline/requirements_offline.txt # pinned deps for offline install
```

---

## Dependencies

Pinned set (see `offline/requirements_offline.txt` for full list and versions):

- **Core:** numpy, pandas, scipy, matplotlib, Pillow, seaborn, requests  
- **UI:** dash, dash-bootstrap-components, plotly  
- **Imaging/CV:** opencv-python, scikit-image, tifffile  
- **ML/Utils:** scikit-learn, ultralytics (utilities)  
- **Torch CPU:** `torch==2.2.2` (ensure correct wheel for your OS/arch)  
- **Jupyter widgets (optional):** ipywidgets, ipycanvas, ipyevents  
- **SAM deps:** onnxruntime, pycocotools  
- **macOS notes:** install Xcode CLI tools if missing for some build steps

**GPU use:** If CUDA is available and compatible wheels are installed, SAM will run on GPU automatically; otherwise CPU.

---

## Common issues

- **Multiple browser windows:** The app runs with `debug=False, use_reloader=False` to avoid multi-process reloads.  
- **Port already in use:** The launcher logic checks port 8050; if occupied, it opens the existing server in the browser.  
- **Missing SAM checkpoints:** Place `sam_vit_*.pth` in the repo root or `cache/`. The app auto-detects them.

---

## Citation and licensing

Copyright © 2025 Harvard University Lichtman Lab.

We welcome community use and modification for research. Please cite our work and acknowledge the HI-MC project:

- Fuming Yang, Lichtman Lab, “WaferTools: Software for High-throughput Integrative Connectomics,” Harvard University, 2025.

**BibTeX example:**
```bibtex
@software{wafertools2025,
  author = {F. Yang and Y. Meirovitch and F. Araújo and R. Schalek and V. Susoy and J.W. Lichtman},
  title  = {WaferTools: Software for High-Throughput Integrative Connectomics},
  year   = {2025},
  url    = {https://github.com/fumingyang-felix/WaferTools},
  urldate= {2025-09-23}
}
```

If your usage is commercial or outside academic research, please contact us.

---

## Contact

Questions, issues, or ideas for improvements are very welcome:

📧 fumingyang@fas.harvard.edu

Contributions (bug reports, PRs) are appreciated — please open an issue or pull request on GitHub with a clear description and reproduction steps.
