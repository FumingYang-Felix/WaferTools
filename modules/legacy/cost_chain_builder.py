#!/usr/bin/env python3
"""cost_chain_builder.py – greedy linear ordering using pairwise cost

This script builds a single linear chain of EM sections using a *greedy* rule
based on a cost function:
    cost = d_um + 100 * (1 - ssim) + 2 * |angle_deg|

Algorithm (nearest-neighbour greedy)
------------------------------------
1. Load CSV (must contain columns: fixed, moving, d_um, ssim, angle_deg).
2. Compute cost for every pair and keep it in a dict.
3. Find the cheapest edge overall – use its two sections as chain seeds
   (left, right = edge[0]).
4. While there are still unvisited sections:
   a) For *each* chain tail (leftmost, rightmost) find its cheapest edge that
      connects to an unvisited section.
   b) Pick the smaller of the two costs and extend the corresponding side.
   c) If neither tail has a valid edge (disconnected set), pick the overall
      cheapest edge among remaining nodes and attach to the *right* side.
5. Output the resulting order and, if matplotlib is available, create a PDF/PNG
   with the per-edge cost curve along the chain.

Usage
-----
python cost_chain_builder.py full_pairwise_rigid_transforms.csv \
       --order-out chain_order_cost.txt --plot metrics_cost.pdf
"""
from __future__ import annotations
import csv, argparse, sys
from pathlib import Path
from typing import Dict, Tuple, List, Set

import math

LAMBDA = 100.0
W_ROT = 2.0
REQ_COLS = ["fixed", "moving", "d_um", "ssim", "angle_deg"]

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# -----------------------------------------------------------------------------

def parse_header(header: List[str]) -> Dict[str, int]:
    idx = {c.lower(): i for i, c in enumerate(header)}
    missing = [c for c in REQ_COLS if c not in idx]
    if missing:
        sys.exit(f"ERROR: required columns not found in CSV header: {', '.join(missing)}")
    return idx


def compute_cost(row: List[str], idx: Dict[str, int]) -> float:
    d = float(row[idx["d_um"]])
    ssim = float(row[idx["ssim"]])
    angle = abs(float(row[idx["angle_deg"]]))
    return d + LAMBDA * (1.0 - ssim) + W_ROT * angle


def load_costs(csv_path: Path) -> Tuple[Dict[Tuple[str, str], float], Set[str]]:
    costs: Dict[Tuple[str, str], float] = {}
    nodes: Set[str] = set()
    with open(csv_path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        idx = parse_header(header)
        for row in rdr:
            if len(row) < len(header):
                continue
            a = row[idx["fixed"]].strip()
            b = row[idx["moving"]].strip()
            if a == b:
                continue
            c = compute_cost(row, idx)
            costs[(a, b)] = c
            costs[(b, a)] = c
            nodes.update([a, b])
    return costs, nodes

# -----------------------------------------------------------------------------
# Greedy linear-order builder
# -----------------------------------------------------------------------------

def greedy_linear_order(costs: Dict[Tuple[str, str], float], nodes: Set[str]):
    # 1) find global cheapest edge
    seed_edge, seed_cost = min(costs.items(), key=lambda kv: kv[1])
    left, right = list(seed_edge)  # tuple order
    chain: List[str] = [left, right]
    visited = {left, right}

    def cheapest_edge_from(node: str):
        cand = [(cost, nb) for (n1, nb), cost in costs.items() if n1 == node and nb not in visited]
        return min(cand, default=(math.inf, None))

    while len(visited) < len(nodes):
        # Check both ends
        left_cost, left_neighbor = cheapest_edge_from(chain[0])
        right_cost, right_neighbor = cheapest_edge_from(chain[-1])

        # Choose better extension
        choices = [(left_cost, "left", left_neighbor), (right_cost, "right", right_neighbor)]
        choices = [c for c in choices if c[2] is not None]

        if choices:
            cost_val, side, nb = min(choices, key=lambda x: x[0])
            if side == "left":
                chain.insert(0, nb)
            else:
                chain.append(nb)
            visited.add(nb)
        else:
            # disconnected – pick cheapest remaining edge overall
            remaining_edges = [(c, a, b) for (a, b), c in costs.items() if a not in visited and b not in visited]
            if not remaining_edges:
                # just append any leftover node
                leftover = next(iter(nodes - visited))
                chain.append(leftover)
                visited.add(leftover)
            else:
                cval, a, b = min(remaining_edges, key=lambda x: x[0])
                chain.append(a)
                chain.append(b)
                visited.update([a, b])
    return chain

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_edge_costs(chain: List[str], costs: Dict[Tuple[str, str], float], out_path: Path):
    if not HAS_PLT:
        print("matplotlib not available – skip plot")
        return
    edge_costs = [costs[(a, b)] for a, b in zip(chain[:-1], chain[1:])]
    x = list(range(1, len(edge_costs) + 1))
    plt.figure(figsize=(8, 3))
    plt.plot(x, edge_costs, marker="o", lw=1)
    plt.xlabel("step")
    plt.ylabel("edge cost")
    plt.title("Greedy linear-order edge costs")
    plt.tight_layout()
    if out_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        plt.savefig(out_path, dpi=150)
    else:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out_path) as pdf:
            pdf.savefig()
    plt.close()
    print("Saved plot ->", out_path)

# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Greedy linear ordering based on cost metric")
    ap.add_argument("csv_in", help="full_pairwise_rigid_transforms.csv")
    ap.add_argument("--order-out", default="chain_order_cost_linear.txt")
    ap.add_argument("--plot", help="output PDF/PNG file for edge-cost metrics")
    args = ap.parse_args()

    costs, nodes = load_costs(Path(args.csv_in))
    print(f"Loaded {len(nodes)} sections, {len(costs)//2} unique pairs")

    chain = greedy_linear_order(costs, nodes)

    # save order
    Path(args.order_out).write_text(" ".join(chain))
    print("Saved order ->", args.order_out)

    # plot metrics if requested
    if args.plot:
        plot_edge_costs(chain, costs, Path(args.plot))

if __name__ == "__main__":
    main() 



 
 
 
 
 
 
 
 
 
 
 
 
 