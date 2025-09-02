#!/usr/bin/env python3
"""extract_ssim_pairs.py – filter pairwise_alignment_results.csv to rows with valid SSIM

Usage
-----
python extract_ssim_pairs.py pairwise_alignment_results.csv --out pair_ssim_only.csv
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description='Filter rows where ssim != -1')
    ap.add_argument('csv_in')
    ap.add_argument('--out', default='pair_ssim_only.csv')
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    rows_kept = 0
    with open(in_path, newline='') as fin, open(args.out, 'w', newline='') as fout:
        rdr = csv.reader(fin)
        wtr = csv.writer(fout)
        header = next(rdr)
        ssim_idx = header.index('ssim') if 'ssim' in header else -2
        ang_idx = header.index('angle_deg') if 'angle_deg' in header else None
        wtr.writerow(header)
        for row in rdr:
            if len(row) <= ssim_idx: continue
            try:
                ssim_val = float(row[ssim_idx])
                ang_val  = float(row[ang_idx]) if ang_idx is not None else 0.0
            except ValueError:
                continue
            if ssim_val != -1.0 and abs(ang_val) <= 10.0:
                wtr.writerow(row);
                rows_kept +=1
    print('Wrote', rows_kept, 'rows ->', args.out)

if __name__ == '__main__':
    main() 
"""extract_ssim_pairs.py – filter pairwise_alignment_results.csv to rows with valid SSIM

Usage
-----
python extract_ssim_pairs.py pairwise_alignment_results.csv --out pair_ssim_only.csv
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description='Filter rows where ssim != -1')
    ap.add_argument('csv_in')
    ap.add_argument('--out', default='pair_ssim_only.csv')
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    rows_kept = 0
    with open(in_path, newline='') as fin, open(args.out, 'w', newline='') as fout:
        rdr = csv.reader(fin)
        wtr = csv.writer(fout)
        header = next(rdr)
        ssim_idx = header.index('ssim') if 'ssim' in header else -2
        ang_idx = header.index('angle_deg') if 'angle_deg' in header else None
        wtr.writerow(header)
        for row in rdr:
            if len(row) <= ssim_idx: continue
            try:
                ssim_val = float(row[ssim_idx])
                ang_val  = float(row[ang_idx]) if ang_idx is not None else 0.0
            except ValueError:
                continue
            if ssim_val != -1.0 and abs(ang_val) <= 10.0:
                wtr.writerow(row);
                rows_kept +=1
    print('Wrote', rows_kept, 'rows ->', args.out)

if __name__ == '__main__':
    main() 
 
"""extract_ssim_pairs.py – filter pairwise_alignment_results.csv to rows with valid SSIM

Usage
-----
python extract_ssim_pairs.py pairwise_alignment_results.csv --out pair_ssim_only.csv
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description='Filter rows where ssim != -1')
    ap.add_argument('csv_in')
    ap.add_argument('--out', default='pair_ssim_only.csv')
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    rows_kept = 0
    with open(in_path, newline='') as fin, open(args.out, 'w', newline='') as fout:
        rdr = csv.reader(fin)
        wtr = csv.writer(fout)
        header = next(rdr)
        ssim_idx = header.index('ssim') if 'ssim' in header else -2
        ang_idx = header.index('angle_deg') if 'angle_deg' in header else None
        wtr.writerow(header)
        for row in rdr:
            if len(row) <= ssim_idx: continue
            try:
                ssim_val = float(row[ssim_idx])
                ang_val  = float(row[ang_idx]) if ang_idx is not None else 0.0
            except ValueError:
                continue
            if ssim_val != -1.0 and abs(ang_val) <= 10.0:
                wtr.writerow(row);
                rows_kept +=1
    print('Wrote', rows_kept, 'rows ->', args.out)

if __name__ == '__main__':
    main() 
"""extract_ssim_pairs.py – filter pairwise_alignment_results.csv to rows with valid SSIM

Usage
-----
python extract_ssim_pairs.py pairwise_alignment_results.csv --out pair_ssim_only.csv
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description='Filter rows where ssim != -1')
    ap.add_argument('csv_in')
    ap.add_argument('--out', default='pair_ssim_only.csv')
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    rows_kept = 0
    with open(in_path, newline='') as fin, open(args.out, 'w', newline='') as fout:
        rdr = csv.reader(fin)
        wtr = csv.writer(fout)
        header = next(rdr)
        ssim_idx = header.index('ssim') if 'ssim' in header else -2
        ang_idx = header.index('angle_deg') if 'angle_deg' in header else None
        wtr.writerow(header)
        for row in rdr:
            if len(row) <= ssim_idx: continue
            try:
                ssim_val = float(row[ssim_idx])
                ang_val  = float(row[ang_idx]) if ang_idx is not None else 0.0
            except ValueError:
                continue
            if ssim_val != -1.0 and abs(ang_val) <= 10.0:
                wtr.writerow(row);
                rows_kept +=1
    print('Wrote', rows_kept, 'rows ->', args.out)

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 