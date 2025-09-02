"""
Section Counter module boundary

Current live files (kept at project root for compatibility):
- section_counter.py (Dash layout + callbacks)
- downsampled_sam.py (SAM detector utilities)
- sam_vit_*.pth (SAM weights)

Planned migration (future, non-breaking):
- Move helper utilities under modules/common and modules/section_counter/helpers.py
- Keep section_counter.py as the Dash-facing entry, importing internal helpers
"""

