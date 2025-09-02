#!/usr/bin/env python3
"""reorder_pair_scores_by_section.py  –  regroup pairwise similarity CSV

Usage
-----
python reorder_pair_scores_by_section.py pair_scores_ncc.csv --out grouped_pair_scores.csv

The script reads a pairwise similarity CSV that at minimum contains the columns
fixed, moving, score (score in the last column). It then:
1. Groups all rows by the *fixed* section name (first column).
2. Inside each group sorts rows by descending score (higher similarity first).
3. Writes the re-ordered rows back to CSV (default: <orig_stem>_grouped.csv).

Any header row from the input is preserved as the first row of the output.
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(x:str)->bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_rows(csv_path:Path)->Tuple[List[str], List[List[str]]]:
    """Return (header_or_None, rows_without_header)."""
    with open(csv_path, newline="") as f:
        rdr=csv.reader(f)
        all_rows=list(rdr)
    if not all_rows:
        return None, []
    # detect header: if last column can't be float
    if not is_float(all_rows[0][-1]):
        header=all_rows[0]
        rows=all_rows[1:]
    else:
        header=None
        rows=all_rows
    return header,rows

def group_and_sort(rows:List[List[str]])->List[List[str]]:
    groups={}
    for row in rows:
        if len(row)<3:  # skip malformed
            continue
        fixed=row[0]
        groups.setdefault(fixed,[]).append(row)
    # build ordered output
    out=[]
    for fixed in sorted(groups.keys()):
        grp=groups[fixed]
        # sort by score descending (last column as float)
        grp.sort(key=lambda r:-float(r[-1]))
        out.extend(grp)
    return out

def main():
    ap=argparse.ArgumentParser(description="Regroup pairwise CSV by fixed section and sort by descending score within each group")
    ap.add_argument('csv_in', help='input pair_scores.csv path')
    ap.add_argument('--out', help='output csv path (default <stem>_grouped.csv)')
    args=ap.parse_args()

    csv_path=Path(args.csv_in)
    header,rows=load_rows(csv_path)
    ordered=group_and_sort(rows)

    out_path=Path(args.out) if args.out else csv_path.with_stem(csv_path.stem+'_grouped')
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(ordered)
    print('Wrote', out_path, 'total rows', len(ordered))

if __name__=='__main__':
    main() 
"""reorder_pair_scores_by_section.py  –  regroup pairwise similarity CSV

Usage
-----
python reorder_pair_scores_by_section.py pair_scores_ncc.csv --out grouped_pair_scores.csv

The script reads a pairwise similarity CSV that at minimum contains the columns
fixed, moving, score (score in the last column). It then:
1. Groups all rows by the *fixed* section name (first column).
2. Inside each group sorts rows by descending score (higher similarity first).
3. Writes the re-ordered rows back to CSV (default: <orig_stem>_grouped.csv).

Any header row from the input is preserved as the first row of the output.
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(x:str)->bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_rows(csv_path:Path)->Tuple[List[str], List[List[str]]]:
    """Return (header_or_None, rows_without_header)."""
    with open(csv_path, newline="") as f:
        rdr=csv.reader(f)
        all_rows=list(rdr)
    if not all_rows:
        return None, []
    # detect header: if last column can't be float
    if not is_float(all_rows[0][-1]):
        header=all_rows[0]
        rows=all_rows[1:]
    else:
        header=None
        rows=all_rows
    return header,rows

def group_and_sort(rows:List[List[str]])->List[List[str]]:
    groups={}
    for row in rows:
        if len(row)<3:  # skip malformed
            continue
        fixed=row[0]
        groups.setdefault(fixed,[]).append(row)
    # build ordered output
    out=[]
    for fixed in sorted(groups.keys()):
        grp=groups[fixed]
        # sort by score descending (last column as float)
        grp.sort(key=lambda r:-float(r[-1]))
        out.extend(grp)
    return out

def main():
    ap=argparse.ArgumentParser(description="Regroup pairwise CSV by fixed section and sort by descending score within each group")
    ap.add_argument('csv_in', help='input pair_scores.csv path')
    ap.add_argument('--out', help='output csv path (default <stem>_grouped.csv)')
    args=ap.parse_args()

    csv_path=Path(args.csv_in)
    header,rows=load_rows(csv_path)
    ordered=group_and_sort(rows)

    out_path=Path(args.out) if args.out else csv_path.with_stem(csv_path.stem+'_grouped')
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(ordered)
    print('Wrote', out_path, 'total rows', len(ordered))

if __name__=='__main__':
    main() 
 
"""reorder_pair_scores_by_section.py  –  regroup pairwise similarity CSV

Usage
-----
python reorder_pair_scores_by_section.py pair_scores_ncc.csv --out grouped_pair_scores.csv

The script reads a pairwise similarity CSV that at minimum contains the columns
fixed, moving, score (score in the last column). It then:
1. Groups all rows by the *fixed* section name (first column).
2. Inside each group sorts rows by descending score (higher similarity first).
3. Writes the re-ordered rows back to CSV (default: <orig_stem>_grouped.csv).

Any header row from the input is preserved as the first row of the output.
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(x:str)->bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_rows(csv_path:Path)->Tuple[List[str], List[List[str]]]:
    """Return (header_or_None, rows_without_header)."""
    with open(csv_path, newline="") as f:
        rdr=csv.reader(f)
        all_rows=list(rdr)
    if not all_rows:
        return None, []
    # detect header: if last column can't be float
    if not is_float(all_rows[0][-1]):
        header=all_rows[0]
        rows=all_rows[1:]
    else:
        header=None
        rows=all_rows
    return header,rows

def group_and_sort(rows:List[List[str]])->List[List[str]]:
    groups={}
    for row in rows:
        if len(row)<3:  # skip malformed
            continue
        fixed=row[0]
        groups.setdefault(fixed,[]).append(row)
    # build ordered output
    out=[]
    for fixed in sorted(groups.keys()):
        grp=groups[fixed]
        # sort by score descending (last column as float)
        grp.sort(key=lambda r:-float(r[-1]))
        out.extend(grp)
    return out

def main():
    ap=argparse.ArgumentParser(description="Regroup pairwise CSV by fixed section and sort by descending score within each group")
    ap.add_argument('csv_in', help='input pair_scores.csv path')
    ap.add_argument('--out', help='output csv path (default <stem>_grouped.csv)')
    args=ap.parse_args()

    csv_path=Path(args.csv_in)
    header,rows=load_rows(csv_path)
    ordered=group_and_sort(rows)

    out_path=Path(args.out) if args.out else csv_path.with_stem(csv_path.stem+'_grouped')
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(ordered)
    print('Wrote', out_path, 'total rows', len(ordered))

if __name__=='__main__':
    main() 
"""reorder_pair_scores_by_section.py  –  regroup pairwise similarity CSV

Usage
-----
python reorder_pair_scores_by_section.py pair_scores_ncc.csv --out grouped_pair_scores.csv

The script reads a pairwise similarity CSV that at minimum contains the columns
fixed, moving, score (score in the last column). It then:
1. Groups all rows by the *fixed* section name (first column).
2. Inside each group sorts rows by descending score (higher similarity first).
3. Writes the re-ordered rows back to CSV (default: <orig_stem>_grouped.csv).

Any header row from the input is preserved as the first row of the output.
"""
from __future__ import annotations
import csv, argparse
from pathlib import Path
from typing import List, Tuple

def is_float(x:str)->bool:
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_rows(csv_path:Path)->Tuple[List[str], List[List[str]]]:
    """Return (header_or_None, rows_without_header)."""
    with open(csv_path, newline="") as f:
        rdr=csv.reader(f)
        all_rows=list(rdr)
    if not all_rows:
        return None, []
    # detect header: if last column can't be float
    if not is_float(all_rows[0][-1]):
        header=all_rows[0]
        rows=all_rows[1:]
    else:
        header=None
        rows=all_rows
    return header,rows

def group_and_sort(rows:List[List[str]])->List[List[str]]:
    groups={}
    for row in rows:
        if len(row)<3:  # skip malformed
            continue
        fixed=row[0]
        groups.setdefault(fixed,[]).append(row)
    # build ordered output
    out=[]
    for fixed in sorted(groups.keys()):
        grp=groups[fixed]
        # sort by score descending (last column as float)
        grp.sort(key=lambda r:-float(r[-1]))
        out.extend(grp)
    return out

def main():
    ap=argparse.ArgumentParser(description="Regroup pairwise CSV by fixed section and sort by descending score within each group")
    ap.add_argument('csv_in', help='input pair_scores.csv path')
    ap.add_argument('--out', help='output csv path (default <stem>_grouped.csv)')
    args=ap.parse_args()

    csv_path=Path(args.csv_in)
    header,rows=load_rows(csv_path)
    ordered=group_and_sort(rows)

    out_path=Path(args.out) if args.out else csv_path.with_stem(csv_path.stem+'_grouped')
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(ordered)
    print('Wrote', out_path, 'total rows', len(ordered))

if __name__=='__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 