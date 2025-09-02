import argparse
import csv
import cv2
from pathlib import Path

def main():
    print('Batch script started')
    ap = argparse.ArgumentParser(description='Batch SIFT alignment for CSV pairs')
    ap.add_argument('csv_in')
    ap.add_argument('--img-dir', default='w7_png_4k')
    ap.add_argument('--out-dir', default='sift_vis_1k')
    ap.add_argument('--max', type=int, default=None, help='max pairs to process')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    with open(args.csv_in) as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            try:
                if args.max and i >= args.max:
                    break
                sec1 = row['fixed']
                sec2 = row['moving']
                print(f'Processing row {i}: {sec1} vs {sec2}')
                img1_path = img_dir / f'{sec1}.png'
                img2_path = img_dir / f'{sec2}.png'
                if not img1_path.exists() or not img2_path.exists():
                    print(f"[skip] {img1_path} or {img2_path} not found")
                    continue
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                if img1 is None or img2 is None:
                    print(f"[skip] failed to load {img1_path} or {img2_path}")
                    continue
                # Run alignment
                results = perform_sift_alignment(img1, img2, sec1, sec2)
                # Save 1k x 1k visualization
                out_path = out_dir / f'sift_{sec1}_vs_{sec2}.png'
                create_visualization(img1, img2, sec1, sec2, results, str(out_path))
                # Resize to 1k x 1k
                img = cv2.imread(str(out_path))
                if img is not None:
                    img_small = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out_path), img_small)
                print(f'[done] {out_path}')
            except Exception as e:
                print(f'[error] row {i}: {e}')

if __name__ == '__main__':
    main() 
import csv
import cv2
from pathlib import Path

def main():
    print('Batch script started')
    ap = argparse.ArgumentParser(description='Batch SIFT alignment for CSV pairs')
    ap.add_argument('csv_in')
    ap.add_argument('--img-dir', default='w7_png_4k')
    ap.add_argument('--out-dir', default='sift_vis_1k')
    ap.add_argument('--max', type=int, default=None, help='max pairs to process')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    with open(args.csv_in) as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            try:
                if args.max and i >= args.max:
                    break
                sec1 = row['fixed']
                sec2 = row['moving']
                print(f'Processing row {i}: {sec1} vs {sec2}')
                img1_path = img_dir / f'{sec1}.png'
                img2_path = img_dir / f'{sec2}.png'
                if not img1_path.exists() or not img2_path.exists():
                    print(f"[skip] {img1_path} or {img2_path} not found")
                    continue
                img1 = cv2.imread(str(img1_path))
                img2 = cv2.imread(str(img2_path))
                if img1 is None or img2 is None:
                    print(f"[skip] failed to load {img1_path} or {img2_path}")
                    continue
                # Run alignment
                results = perform_sift_alignment(img1, img2, sec1, sec2)
                # Save 1k x 1k visualization
                out_path = out_dir / f'sift_{sec1}_vs_{sec2}.png'
                create_visualization(img1, img2, sec1, sec2, results, str(out_path))
                # Resize to 1k x 1k
                img = cv2.imread(str(out_path))
                if img is not None:
                    img_small = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out_path), img_small)
                print(f'[done] {out_path}')
            except Exception as e:
                print(f'[error] row {i}: {e}')

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 