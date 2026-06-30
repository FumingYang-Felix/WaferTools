#!/usr/bin/env python3
"""
png_compress.py — compress PNG images with optional resizing and color quantization.

Usage examples:
  # Lossless-ish (optimize zlib & strip metadata)
  python png_compress.py input.png -o output.png

  # Stronger compression: quantize to 128 colors (slightly lossy), remove metadata
  python png_compress.py input.png -o output.png --colors 128

  # Resize (max width/height) and compress
  python png_compress.py input.png -o output.png --max-width 1600 --max-height 1600 --colors 128

  # In-place (only overwrite if the result is smaller)
  python png_compress.py input.png --inplace

Requires:
  pip install pillow
"""
import argparse
import os
from io import BytesIO
from PIL import Image, PngImagePlugin, ImageOps

def human_size(n: int) -> str:
    for unit in ("B","KB","MB","GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.2f}{unit}" if unit in ("KB","MB","GB") else f"{n}{unit}"
        n /= 1024
    return f"{n:.2f}GB"

def compress_png(
    in_path: str,
    out_path: str | None,
    inplace: bool,
    max_width: int | None,
    max_height: int | None,
    colors: int | None,
    dither: bool,
    posterize_bits: int | None,
    zlib_level: int,
    keep_metadata: bool,
    interlace: bool,
) -> str:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    orig_size = os.path.getsize(in_path)

    with Image.open(in_path) as im:
        im.load()  # ensure data is read before operations

        # Optional resize while preserving aspect ratio
        if max_width or max_height:
            # Compute bounding box
            max_w = max_width or im.width
            max_h = max_height or im.height
            im = ImageOps.exif_transpose(im)  # respect orientation
            im.thumbnail((max_w, max_h), Image.LANCZOS)

        # Optional posterize (reduces per-channel bit depth; mild quality loss)
        if posterize_bits is not None:
            if im.mode not in ("RGB", "L"):
                # Convert to RGB safely (keeps alpha in separate channel if present)
                if "A" in im.getbands():
                    # Split alpha, posterize RGB, then recombine
                    rgb = im.convert("RGB")
                    alpha = im.getchannel("A")
                    rgb = ImageOps.posterize(rgb, posterize_bits)
                    im = Image.merge("RGBA", (*rgb.split(), alpha))
                else:
                    im = ImageOps.posterize(im.convert("RGB"), posterize_bits)
            else:
                im = ImageOps.posterize(im, posterize_bits)

        # Optional quantization (biggest size savings). PNG-8 palette with up to 256 colors.
        if colors is not None:
            # Ensure valid range
            colors = max(2, min(256, colors))
            # Dithering: FLOYDSTEINBERG=1, NONE=0
            dither_mode = Image.FLOYDSTEINBERG if dither else Image.NONE
            # If image has alpha, Pillow can quantize with alpha as palette transparency.
            im = im.quantize(colors=colors, method=Image.FASTOCTREE, dither=dither_mode)

        # Prepare metadata handling
        save_kwargs = {
            "optimize": True,                 # better DEFLATE tables
            "compress_level": int(zlib_level),# 0..9 (9 is smallest/slowest)
            "interlace": int(interlace),      # 0 or 1
        }

        pnginfo = None
        if keep_metadata:
            pnginfo = PngImagePlugin.PngInfo()
            # Copy text-based metadata only (avoid bloating/invalid bytes)
            for k, v in (getattr(im, "info", {}) or {}).items():
                if isinstance(v, str):
                    try:
                        pnginfo.add_text(k, v)
                    except Exception:
                        pass
            # Preserve ICC if present
            if "icc_profile" in im.info:
                save_kwargs["icc_profile"] = im.info["icc_profile"]
        # Else: strip metadata by not passing pnginfo/icc_profile

        # Save to an in-memory buffer first so we can compare sizes
        buf = BytesIO()
        im.save(buf, format="PNG", pnginfo=pnginfo, **save_kwargs)
        data = buf.getvalue()
        new_size = len(data)

    # Decide output path
    if out_path is None:
        if inplace:
            out_path = in_path
        else:
            root, ext = os.path.splitext(in_path)
            out_path = f"{root}.compressed.png"

    # Only overwrite in-place if smaller (safety)
    if inplace and new_size >= orig_size:
        msg = (f"No in-place overwrite: compressed file would be larger "
               f"({human_size(new_size)} vs {human_size(orig_size)}). Kept original.")
        print(msg)
        return in_path

    # Write file
    with open(out_path, "wb") as f:
        f.write(data)

    # Report
    saved = orig_size - new_size
    pct = (saved / orig_size * 100) if orig_size else 0.0
    print(f"Original:  {human_size(orig_size)}")
    print(f"Compressed:{human_size(new_size)}")
    if saved >= 0:
        print(f"Saved:    {human_size(saved)} ({pct:.1f}%)")
    else:
        print(f"Warning: file grew by {human_size(-saved)} ({-pct:.1f}%).")
    print(f"Wrote:     {out_path}")
    return out_path

def parse_args():
    p = argparse.ArgumentParser(description="Compress PNG images with optional resizing and quantization.")
    p.add_argument("input", help="Path to input PNG")
    p.add_argument("-o", "--output", help="Path to output PNG (default: <input>.compressed.png or --inplace)")
    p.add_argument("--inplace", action="store_true", help="Overwrite input only if the result is smaller")
    p.add_argument("--max-width", type=int, help="Max width in pixels (maintains aspect ratio)")
    p.add_argument("--max-height", type=int, help="Max height in pixels (maintains aspect ratio)")
    p.add_argument("--colors", type=int, help="Quantize to this many colors (2..256). Greatly reduces size (lossy).")
    p.add_argument("--no-dither", action="store_true", help="Disable dithering when quantizing (may reduce size)")
    p.add_argument("--posterize-bits", type=int, choices=range(1,9),
                   help="Reduce per-channel bit depth to N bits (1-8). Mild lossy. Often pairs with --colors.")
    p.add_argument("--zlib-level", type=int, default=9, choices=range(0,10),
                   help="PNG DEFLATE level 0..9 (default: 9 smallest/slowest)")
    p.add_argument("--keep-metadata", action="store_true", help="Preserve textual metadata and ICC profile")
    p.add_argument("--interlace", action="store_true", help="Enable Adam7 interlacing (usually larger files)")
    return p.parse_args()

def main():
    args = parse_args()
    compress_png(
        in_path=args.input,
        out_path=args.output,
        inplace=args.inplace,
        max_width=args.max_width,
        max_height=args.max_height,
        colors=args.colors,
        dither=not args.no_dither,
        posterize_bits=args.posterize_bits,
        zlib_level=args.zlib_level,
        keep_metadata=args.keep_metadata,
        interlace=args.interlace,
    )

if __name__ == "__main__":
    main()
