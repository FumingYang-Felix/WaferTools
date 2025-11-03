#!/usr/bin/env python3
"""
resize_png.py — directly resize (downscale) large images and save as PNG.

Examples:
  # Fit within 8000 px tall (keep aspect)
  python resize_png.py input.png -o output.png --max-height 8000

  # Fit within a 8000 x 8000 box
  python resize_png.py input.png -o output.png --max-width 8000 --max-height 8000

  # Set exact width (height scales)
  python resize_png.py input.png -o output.png --width 4000

  # In-place (only overwrites if smaller OR if --force is given)
  python resize_png.py input.png --inplace --max-height 6000
"""
import argparse
import os
from io import BytesIO
from PIL import Image, ImageOps, ImageFile

# Allow very large images to load (you're intentionally resizing them)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly damaged files

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"

def compute_target_size(
    w: int, h: int,
    width: int | None,
    height: int | None,
    max_width: int | None,
    max_height: int | None,
    allow_upscale: bool
) -> tuple[int, int]:
    if width and height:
        # Treat as a bounding box: fit inside width x height
        scale = min(width / w, height / h)
    elif width:
        scale = width / w
    elif height:
        scale = height / h
    elif max_width or max_height:
        mw = max_width or w
        mh = max_height or h
        scale = min(mw / w, mh / h)
    else:
        raise ValueError("Specify one of: --width/--height or --max-width/--max-height")

    if not allow_upscale:
        scale = min(scale, 1.0)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return new_w, new_h

def resize_to_png(
    in_path: str,
    out_path: str | None,
    inplace: bool,
    width: int | None,
    height: int | None,
    max_width: int | None,
    max_height: int | None,
    allow_upscale: bool,
    zlib_level: int,
    interlace: bool,
    force_overwrite: bool
) -> str:
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    orig_size = os.path.getsize(in_path)

    with Image.open(in_path) as im:
        im.load()  # read data
        im = ImageOps.exif_transpose(im)  # respect orientation if present

        target_w, target_h = compute_target_size(
            im.width, im.height, width, height, max_width, max_height, allow_upscale
        )

        if (target_w, target_h) == (im.width, im.height):
            print("Target size equals original; re-saving as optimized PNG.")
            resized = im
        else:
            # Use high-quality downscale
            resized = im.resize((target_w, target_h), resample=Image.LANCZOS)

        # Always save as PNG (optimize + high compression). No metadata passed = stripped.
        save_kwargs = {
            "optimize": True,
            "compress_level": int(zlib_level),  # 0..9 (9=smallest)
            "interlace": int(interlace),        # 0/1
        }

        # Write to buffer first to check size
        buf = BytesIO()
        resized.save(buf, format="PNG", **save_kwargs)
        data = buf.getvalue()

    # Decide output path
    if out_path is None:
        if inplace:
            out_path = in_path
        else:
            root, _ = os.path.splitext(in_path)
            out_path = f"{root}.resized.png"

    new_size = len(data)

    # In-place safety: only overwrite if smaller or forcing
    if inplace and not force_overwrite and new_size >= orig_size:
        print(
            f"Skip overwrite: resized file would be larger "
            f"({human_size(new_size)} vs {human_size(orig_size)}). Kept original."
        )
        return in_path

    with open(out_path, "wb") as f:
        f.write(data)

    saved = orig_size - new_size
    pct = (saved / orig_size * 100) if orig_size else 0.0
    print(f"Original:   {human_size(orig_size)}  ({in_path})")
    print(f"Resized:    {human_size(new_size)}  ({out_path})")
    if saved >= 0:
        print(f"Saved:      {human_size(saved)} ({pct:.1f}%)")
    else:
        print(f"Note: file grew by {human_size(-saved)} ({-pct:.1f}%).")
    print(f"Dimensions: {target_w} x {target_h}")
    return out_path

def parse_args():
    p = argparse.ArgumentParser(description="Directly resize large images and save as optimized PNG.")
    p.add_argument("input", help="Path to input image (PNG/JPEG/TIFF, etc.)")
    p.add_argument("-o", "--output", help="Path to output PNG (default: <input>.resized.png or --inplace)")
    p.add_argument("--inplace", action="store_true", help="Overwrite input if result is smaller (or --force)")
    # Size controls (choose one style)
    p.add_argument("--width", type=int, help="Exact width (height scales to keep aspect)")
    p.add_argument("--height", type=int, help="Exact height (width scales to keep aspect)")
    p.add_argument("--max-width", type=int, help="Max width (fit inside box)")
    p.add_argument("--max-height", type=int, help="Max height (fit inside box)")
    p.add_argument("--upscale", action="store_true", help="Allow upscaling if target larger than original")
    # PNG save knobs
    p.add_argument("--zlib-level", type=int, default=9, choices=range(0, 10),
                   help="PNG DEFLATE compression level 0..9 (default: 9)")
    p.add_argument("--interlace", action="store_true", help="Enable Adam7 interlacing (usually larger files)")
    p.add_argument("--force", action="store_true", help="With --inplace, overwrite even if output is larger")
    return p.parse_args()

def main():
    args = parse_args()
    resize_to_png(
        in_path=args.input,
        out_path=args.output,
        inplace=args.inplace,
        width=args.width,
        height=args.height,
        max_width=args.max_width,
        max_height=args.max_height,
        allow_upscale=args.upscale,
        zlib_level=args.zlib_level,
        interlace=args.interlace,
        force_overwrite=args.force,
    )

if __name__ == "__main__":
    main()
