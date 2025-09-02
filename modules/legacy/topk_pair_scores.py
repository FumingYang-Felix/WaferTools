#!/usr/bin/env python3
"""topk_pair_scores.py – extract top-K neighbours for each section from pairwise CSV.

Usage
-----
python topk_pair_scores.py pair_scores_ncc.csv --k 3 --out pair_scores_top3.csv

• Reads a CSV whose last column is the similarity score.
• Groups by the *fixed* section (column 0) and keeps the top-K rows inside each group
  (sorted by descending score).
• Writes the subset to a new CSV, preserving the header if present and appending an
  extra column `rank` (1..K).
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_csv(path: Path) -> Tuple[List[str] | None, List[List[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    header = None
    if not is_float(rows[0][-1]):
        header, rows = rows[0], rows[1:]
    return header, rows

def topk_rows(rows: List[List[str]], k: int, ascending: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        groups.setdefault(row[0], []).append(row)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort by score or cost
        if ascending:
            grp.sort(key=lambda r: float(r[-1]))  # lower cost first
        else:
            grp.sort(key=lambda r: -float(r[-1]))  # higher score first
        for rank, r in enumerate(grp[:k], 1):
            subset.append(r + [str(rank)])
    return subset

def main():
    ap = argparse.ArgumentParser(description="Extract top-K pairwise scores per section from CSV")
    ap.add_argument("csv_in", help="input pair_scores.csv path")
    ap.add_argument("--k", type=int, default=3, help="number of top pairs to keep per section (default 3)")
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--ascending", action="store_true", help="sort ascending (lower is better, e.g. cost metric)")
    args = ap.parse_args()

    csv_path = Path(args.csv_in)
    header, rows = read_csv(csv_path)
    subset = topk_rows(rows, args.k, args.ascending)

    out_path = Path(args.out) if args.out else csv_path.with_stem(f"{csv_path.stem}_top{args.k}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header + ["rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
"""topk_pair_scores.py – extract top-K neighbours for each section from pairwise CSV.

Usage
-----
python topk_pair_scores.py pair_scores_ncc.csv --k 3 --out pair_scores_top3.csv

• Reads a CSV whose last column is the similarity score.
• Groups by the *fixed* section (column 0) and keeps the top-K rows inside each group
  (sorted by descending score).
• Writes the subset to a new CSV, preserving the header if present and appending an
  extra column `rank` (1..K).
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_csv(path: Path) -> Tuple[List[str] | None, List[List[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    header = None
    if not is_float(rows[0][-1]):
        header, rows = rows[0], rows[1:]
    return header, rows

def topk_rows(rows: List[List[str]], k: int, ascending: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        groups.setdefault(row[0], []).append(row)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort by score or cost
        if ascending:
            grp.sort(key=lambda r: float(r[-1]))  # lower cost first
        else:
            grp.sort(key=lambda r: -float(r[-1]))  # higher score first
        for rank, r in enumerate(grp[:k], 1):
            subset.append(r + [str(rank)])
    return subset

def main():
    ap = argparse.ArgumentParser(description="Extract top-K pairwise scores per section from CSV")
    ap.add_argument("csv_in", help="input pair_scores.csv path")
    ap.add_argument("--k", type=int, default=3, help="number of top pairs to keep per section (default 3)")
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--ascending", action="store_true", help="sort ascending (lower is better, e.g. cost metric)")
    args = ap.parse_args()

    csv_path = Path(args.csv_in)
    header, rows = read_csv(csv_path)
    subset = topk_rows(rows, args.k, args.ascending)

    out_path = Path(args.out) if args.out else csv_path.with_stem(f"{csv_path.stem}_top{args.k}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header + ["rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
 
"""topk_pair_scores.py – extract top-K neighbours for each section from pairwise CSV.

Usage
-----
python topk_pair_scores.py pair_scores_ncc.csv --k 3 --out pair_scores_top3.csv

• Reads a CSV whose last column is the similarity score.
• Groups by the *fixed* section (column 0) and keeps the top-K rows inside each group
  (sorted by descending score).
• Writes the subset to a new CSV, preserving the header if present and appending an
  extra column `rank` (1..K).
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_csv(path: Path) -> Tuple[List[str] | None, List[List[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    header = None
    if not is_float(rows[0][-1]):
        header, rows = rows[0], rows[1:]
    return header, rows

def topk_rows(rows: List[List[str]], k: int, ascending: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        groups.setdefault(row[0], []).append(row)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort by score or cost
        if ascending:
            grp.sort(key=lambda r: float(r[-1]))  # lower cost first
        else:
            grp.sort(key=lambda r: -float(r[-1]))  # higher score first
        for rank, r in enumerate(grp[:k], 1):
            subset.append(r + [str(rank)])
    return subset

def main():
    ap = argparse.ArgumentParser(description="Extract top-K pairwise scores per section from CSV")
    ap.add_argument("csv_in", help="input pair_scores.csv path")
    ap.add_argument("--k", type=int, default=3, help="number of top pairs to keep per section (default 3)")
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--ascending", action="store_true", help="sort ascending (lower is better, e.g. cost metric)")
    args = ap.parse_args()

    csv_path = Path(args.csv_in)
    header, rows = read_csv(csv_path)
    subset = topk_rows(rows, args.k, args.ascending)

    out_path = Path(args.out) if args.out else csv_path.with_stem(f"{csv_path.stem}_top{args.k}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header + ["rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
"""topk_pair_scores.py – extract top-K neighbours for each section from pairwise CSV.

Usage
-----
python topk_pair_scores.py pair_scores_ncc.csv --k 3 --out pair_scores_top3.csv

• Reads a CSV whose last column is the similarity score.
• Groups by the *fixed* section (column 0) and keeps the top-K rows inside each group
  (sorted by descending score).
• Writes the subset to a new CSV, preserving the header if present and appending an
  extra column `rank` (1..K).
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_csv(path: Path) -> Tuple[List[str] | None, List[List[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, []
    header = None
    if not is_float(rows[0][-1]):
        header, rows = rows[0], rows[1:]
    return header, rows

def topk_rows(rows: List[List[str]], k: int, ascending: bool) -> List[List[str]]:
    groups = {}
    for row in rows:
        if len(row) < 3:
            continue
        groups.setdefault(row[0], []).append(row)
    subset: List[List[str]] = []
    for fixed, grp in groups.items():
        # sort by score or cost
        if ascending:
            grp.sort(key=lambda r: float(r[-1]))  # lower cost first
        else:
            grp.sort(key=lambda r: -float(r[-1]))  # higher score first
        for rank, r in enumerate(grp[:k], 1):
            subset.append(r + [str(rank)])
    return subset

def main():
    ap = argparse.ArgumentParser(description="Extract top-K pairwise scores per section from CSV")
    ap.add_argument("csv_in", help="input pair_scores.csv path")
    ap.add_argument("--k", type=int, default=3, help="number of top pairs to keep per section (default 3)")
    ap.add_argument("--out", help="output csv path (default <stem>_topK.csv)")
    ap.add_argument("--ascending", action="store_true", help="sort ascending (lower is better, e.g. cost metric)")
    args = ap.parse_args()

    csv_path = Path(args.csv_in)
    header, rows = read_csv(csv_path)
    subset = topk_rows(rows, args.k, args.ascending)

    out_path = Path(args.out) if args.out else csv_path.with_stem(f"{csv_path.stem}_top{args.k}")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header + ["rank"])
        w.writerows(subset)
    print("Wrote", out_path, "rows", len(subset))

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 