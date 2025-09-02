#!/usr/bin/env python3
"""cost_pairs_sorted.py – compute cost for every pair and sort groups by ascending cost.

Usage
-----
python cost_pairs_sorted.py full_pairwise_rigid_transforms.csv --out transforms_cost_sorted.csv

• Reads CSV with required columns: fixed, moving, d_um, angle_deg, ssim.
• Computes cost = d + 100*(1-ssim) + 2*|angle_deg| for every row.
• Groups rows by fixed section and sorts each group by ascending cost.
• Writes full dataset with new 'cost' column, preserving header, to output CSV.
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

LAMBDA = 100.0
W_ROT = 2.0
REQ_COLS = ["fixed", "moving", "d_um", "angle_deg", "ssim"]

def parse_header(header: List[str]) -> Dict[str,int]:
    mapping = {c.lower(): i for i,c in enumerate(header)}
    for col in REQ_COLS:
        if col not in mapping:
            sys.exit(f"ERROR: column '{col}' not found in CSV header")
    return mapping

def compute_cost(row: List[str], idx: Dict[str,int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    ang = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA*(1-ssim) + W_ROT*ang

def main():
    ap = argparse.ArgumentParser(description="Compute cost and sort pairs by section ascending cost")
    ap.add_argument('csv_in', help='input CSV with transforms')
    ap.add_argument('--out', help='output CSV path (default <stem>_cost_sorted.csv)')
    args=ap.parse_args()
    in_path=Path(args.csv_in)
    with open(in_path, newline='') as f:
        rdr=csv.reader(f)
        header=next(rdr)
        idx=parse_header(header)
        rows=[list(r) for r in rdr]
    # compute cost per row
    rows_with_cost=[]
    for r in rows:
        c=compute_cost(r, idx)
        rows_with_cost.append((r[idx['fixed']], c, r))
    # group and sort
    grouped: Dict[str,List[Tuple[float,List[str]]]]={}
    for fixed,cost_val,row in rows_with_cost:
        grouped.setdefault(fixed,[]).append((cost_val,row))
    ordered_rows: List[List[str]]=[]
    for fixed in sorted(grouped.keys()):
        grp=grouped[fixed]
        grp.sort(key=lambda t:t[0])  # ascending cost
        for cost_val,row in grp:
            ordered_rows.append(row + [f"{cost_val:.6f}"])
    out_path=Path(args.out) if args.out else in_path.with_stem(f"{in_path.stem}_cost_sorted")
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(header+["cost"])
        w.writerows(ordered_rows)
    print('Wrote', out_path, 'rows', len(ordered_rows))

if __name__=='__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 