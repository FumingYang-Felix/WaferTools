#!/usr/bin/env python3
"""topk_cost_pairs.py – compute cost = d + 100*(1-ssim) + 2*|angle| and keep top-K (lowest cost) per section.

Usage
-----
python topk_cost_pairs.py full_pairwise_rigid_transforms.csv --k 3 --out transforms_top3.csv

CSV must contain at least these columns:
  fixed, moving, d_um, angle_deg, ssim
The header can be in any order; the script locates columns by name (case-insensitive).
Output file keeps原有列并附加 cost 与 rank 字段。
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

LAMBDA = 100.0
W_ROT = 2.0

REQ_COLS = ["fixed", "moving", "d_um", "angle_deg", "ssim"]

def parse_header(header: List[str]) -> Dict[str, int]:
    mapping = {c.lower(): i for i, c in enumerate(header)}
    for col in REQ_COLS:
        if col not in mapping:
            sys.exit(f"ERROR: Required column '{col}' not found in CSV header")
    return mapping

def compute_cost(row: List[str], idx: Dict[str, int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    angle = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA * (1.0 - ssim) + W_ROT * angle

def load_rows(path: Path) -> Tuple[List[str], List[List[str]], Dict[str,int]]:
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = parse_header(header)
        rows = [list(r) for r in rdr]
    return header, rows, idx

def topk_rows(rows: List[List[str]], k: int, symmetric: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        fixed, moving = row[0], row[1]
        groups.setdefault(fixed, []).append(row)
        if symmetric:
            # create a shallow copy with fixed/moving swapped
            row_rev = row.copy()
            row_rev[0], row_rev[1] = moving, fixed
            groups.setdefault(moving, []).append(row_rev)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort ascending by cost (last column)
        grp.sort(key=lambda r: float(r[-1]))
        for rank, r in enumerate(grp[:k], start=1):
            r_with = r + [str(rank)]
            subset.append(r_with)
    return subset

def main():
    ap = argparse.ArgumentParser(description="Select top-K minimal cost pairs per section from rigid transforms CSV")
    ap.add_argument("csv_in", help="input full_pairwise_rigid_transforms.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--symmetric", action="store_true", help="treat pairs as undirected (add reversed entries)")
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    header, rows, idx = load_rows(in_path)

    # compute cost and append as last column
    rows_with_cost: List[List[str]] = []
    for r in rows:
        c = compute_cost(r, idx)
        rows_with_cost.append(r + [f"{c:.6f}"])

    subset = topk_rows(rows_with_cost, args.k, args.symmetric)

    out_path = Path(args.out) if args.out else in_path.with_stem(f"{in_path.stem}_top{args.k}_cost")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["cost", "rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
"""topk_cost_pairs.py – compute cost = d + 100*(1-ssim) + 2*|angle| and keep top-K (lowest cost) per section.

Usage
-----
python topk_cost_pairs.py full_pairwise_rigid_transforms.csv --k 3 --out transforms_top3.csv

CSV must contain at least these columns:
  fixed, moving, d_um, angle_deg, ssim
The header can be in any order; the script locates columns by name (case-insensitive).
Output file keeps原有列并附加 cost 与 rank 字段。
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

LAMBDA = 100.0
W_ROT = 2.0

REQ_COLS = ["fixed", "moving", "d_um", "angle_deg", "ssim"]

def parse_header(header: List[str]) -> Dict[str, int]:
    mapping = {c.lower(): i for i, c in enumerate(header)}
    for col in REQ_COLS:
        if col not in mapping:
            sys.exit(f"ERROR: Required column '{col}' not found in CSV header")
    return mapping

def compute_cost(row: List[str], idx: Dict[str, int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    angle = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA * (1.0 - ssim) + W_ROT * angle

def load_rows(path: Path) -> Tuple[List[str], List[List[str]], Dict[str,int]]:
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = parse_header(header)
        rows = [list(r) for r in rdr]
    return header, rows, idx

def topk_rows(rows: List[List[str]], k: int, symmetric: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        fixed, moving = row[0], row[1]
        groups.setdefault(fixed, []).append(row)
        if symmetric:
            # create a shallow copy with fixed/moving swapped
            row_rev = row.copy()
            row_rev[0], row_rev[1] = moving, fixed
            groups.setdefault(moving, []).append(row_rev)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort ascending by cost (last column)
        grp.sort(key=lambda r: float(r[-1]))
        for rank, r in enumerate(grp[:k], start=1):
            r_with = r + [str(rank)]
            subset.append(r_with)
    return subset

def main():
    ap = argparse.ArgumentParser(description="Select top-K minimal cost pairs per section from rigid transforms CSV")
    ap.add_argument("csv_in", help="input full_pairwise_rigid_transforms.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--symmetric", action="store_true", help="treat pairs as undirected (add reversed entries)")
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    header, rows, idx = load_rows(in_path)

    # compute cost and append as last column
    rows_with_cost: List[List[str]] = []
    for r in rows:
        c = compute_cost(r, idx)
        rows_with_cost.append(r + [f"{c:.6f}"])

    subset = topk_rows(rows_with_cost, args.k, args.symmetric)

    out_path = Path(args.out) if args.out else in_path.with_stem(f"{in_path.stem}_top{args.k}_cost")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["cost", "rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
 
"""topk_cost_pairs.py – compute cost = d + 100*(1-ssim) + 2*|angle| and keep top-K (lowest cost) per section.

Usage
-----
python topk_cost_pairs.py full_pairwise_rigid_transforms.csv --k 3 --out transforms_top3.csv

CSV must contain at least these columns:
  fixed, moving, d_um, angle_deg, ssim
The header can be in any order; the script locates columns by name (case-insensitive).
Output file keeps原有列并附加 cost 与 rank 字段。
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

LAMBDA = 100.0
W_ROT = 2.0

REQ_COLS = ["fixed", "moving", "d_um", "angle_deg", "ssim"]

def parse_header(header: List[str]) -> Dict[str, int]:
    mapping = {c.lower(): i for i, c in enumerate(header)}
    for col in REQ_COLS:
        if col not in mapping:
            sys.exit(f"ERROR: Required column '{col}' not found in CSV header")
    return mapping

def compute_cost(row: List[str], idx: Dict[str, int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    angle = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA * (1.0 - ssim) + W_ROT * angle

def load_rows(path: Path) -> Tuple[List[str], List[List[str]], Dict[str,int]]:
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = parse_header(header)
        rows = [list(r) for r in rdr]
    return header, rows, idx

def topk_rows(rows: List[List[str]], k: int, symmetric: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        fixed, moving = row[0], row[1]
        groups.setdefault(fixed, []).append(row)
        if symmetric:
            # create a shallow copy with fixed/moving swapped
            row_rev = row.copy()
            row_rev[0], row_rev[1] = moving, fixed
            groups.setdefault(moving, []).append(row_rev)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort ascending by cost (last column)
        grp.sort(key=lambda r: float(r[-1]))
        for rank, r in enumerate(grp[:k], start=1):
            r_with = r + [str(rank)]
            subset.append(r_with)
    return subset

def main():
    ap = argparse.ArgumentParser(description="Select top-K minimal cost pairs per section from rigid transforms CSV")
    ap.add_argument("csv_in", help="input full_pairwise_rigid_transforms.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--symmetric", action="store_true", help="treat pairs as undirected (add reversed entries)")
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    header, rows, idx = load_rows(in_path)

    # compute cost and append as last column
    rows_with_cost: List[List[str]] = []
    for r in rows:
        c = compute_cost(r, idx)
        rows_with_cost.append(r + [f"{c:.6f}"])

    subset = topk_rows(rows_with_cost, args.k, args.symmetric)

    out_path = Path(args.out) if args.out else in_path.with_stem(f"{in_path.stem}_top{args.k}_cost")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["cost", "rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
"""topk_cost_pairs.py – compute cost = d + 100*(1-ssim) + 2*|angle| and keep top-K (lowest cost) per section.

Usage
-----
python topk_cost_pairs.py full_pairwise_rigid_transforms.csv --k 3 --out transforms_top3.csv

CSV must contain at least these columns:
  fixed, moving, d_um, angle_deg, ssim
The header can be in any order; the script locates columns by name (case-insensitive).
Output file keeps原有列并附加 cost 与 rank 字段。
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import List, Dict, Tuple

LAMBDA = 100.0
W_ROT = 2.0

REQ_COLS = ["fixed", "moving", "d_um", "angle_deg", "ssim"]

def parse_header(header: List[str]) -> Dict[str, int]:
    mapping = {c.lower(): i for i, c in enumerate(header)}
    for col in REQ_COLS:
        if col not in mapping:
            sys.exit(f"ERROR: Required column '{col}' not found in CSV header")
    return mapping

def compute_cost(row: List[str], idx: Dict[str, int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    angle = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA * (1.0 - ssim) + W_ROT * angle

def load_rows(path: Path) -> Tuple[List[str], List[List[str]], Dict[str,int]]:
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = parse_header(header)
        rows = [list(r) for r in rdr]
    return header, rows, idx

def topk_rows(rows: List[List[str]], k: int, symmetric: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        fixed, moving = row[0], row[1]
        groups.setdefault(fixed, []).append(row)
        if symmetric:
            # create a shallow copy with fixed/moving swapped
            row_rev = row.copy()
            row_rev[0], row_rev[1] = moving, fixed
            groups.setdefault(moving, []).append(row_rev)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort ascending by cost (last column)
        grp.sort(key=lambda r: float(r[-1]))
        for rank, r in enumerate(grp[:k], start=1):
            r_with = r + [str(rank)]
            subset.append(r_with)
    return subset

def main():
    ap = argparse.ArgumentParser(description="Select top-K minimal cost pairs per section from rigid transforms CSV")
    ap.add_argument("csv_in", help="input full_pairwise_rigid_transforms.csv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--symmetric", action="store_true", help="treat pairs as undirected (add reversed entries)")
    args = ap.parse_args()

    in_path = Path(args.csv_in)
    header, rows, idx = load_rows(in_path)

    # compute cost and append as last column
    rows_with_cost: List[List[str]] = []
    for r in rows:
        c = compute_cost(r, idx)
        rows_with_cost.append(r + [f"{c:.6f}"])

    subset = topk_rows(rows_with_cost, args.k, args.symmetric)

    out_path = Path(args.out) if args.out else in_path.with_stem(f"{in_path.stem}_top{args.k}_cost")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["cost", "rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 