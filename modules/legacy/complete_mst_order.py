#!/usr/bin/env python3
"""
complete_mst_order.py - Complete graph MST-based EM section ordering
===================================================================
Uses complete graph with MST to find optimal ordering
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import csv
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def load_pair_scores(csv_path: Path) -> Tuple[Dict[Tuple[str,str], float], List[str]]:
    """Load pairwise similarity CSV and convert to distances."""
    scores = {}
    sections = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 9:
                try:
                    sec1, sec2 = row[0].strip(), row[1].strip()
                    ssim_score = float(row[8])  # SSIM is in column 9 (index 8)
                    
                    # Convert to distance (1 - SSIM)
                    distance = 1.0 - ssim_score
                    
                    scores[(sec1, sec2)] = distance
                    scores[(sec2, sec1)] = distance
                    sections.add(sec1)
                    sections.add(sec2)
                except (ValueError, IndexError):
                    continue
    
    return scores, sorted(list(sections))

def complete_mst_ordering(scores: Dict[Tuple[str,str], float], sections: List[str]) -> List[str]:
    """Complete graph MST-based ordering."""
    print("Running complete graph MST ordering...")
    
    # Create complete graph with all possible connections
    G = nx.Graph()
    
    # Add all nodes
    for section in sections:
        G.add_node(section)
    
    # Add all possible edges with distances
    for i, sec1 in enumerate(sections):
        for j, sec2 in enumerate(sections):
            if i != j:
                distance = scores.get((sec1, sec2), 1.0)  # Default distance for missing pairs
                G.add_edge(sec1, sec2, weight=distance)
    
    # Find MST
    mst = nx.minimum_spanning_tree(G, weight='weight')
    
    # Find longest path in MST
    longest_path = []
    max_length = 0
    
    # Try different start and end points
    nodes = list(mst.nodes())
    for start in nodes:
        for end in nodes:
            if start != end:
                try:
                    path = nx.shortest_path(mst, start, end, weight='weight')
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
                except nx.NetworkXNoPath:
                    continue
    
    if not longest_path:
        # Fallback: use any path
        longest_path = list(sections)
    
    print(f"Complete MST ordering complete. Path length: {len(longest_path)}")
    return longest_path

def mst_with_connectivity(scores: Dict[Tuple[str,str], float], sections: List[str]) -> List[str]:
    """MST-based ordering ensuring connectivity."""
    print("Running MST with connectivity...")
    
    # Create graph with existing connections
    G = nx.Graph()
    
    # Add nodes
    for section in sections:
        G.add_node(section)
    
    # Add edges with distances
    for (sec1, sec2), distance in scores.items():
        if sec1 in sections and sec2 in sections:
            G.add_edge(sec1, sec2, weight=distance)
    
    # Check connectivity
    if not nx.is_connected(G):
        print("Graph is not connected. Adding missing edges...")
        # Add missing edges with maximum distance
        max_distance = max(scores.values()) if scores else 1.0
        for i, sec1 in enumerate(sections):
            for j, sec2 in enumerate(sections):
                if i != j and not G.has_edge(sec1, sec2):
                    G.add_edge(sec1, sec2, weight=max_distance * 2)
    
    # Find MST
    mst = nx.minimum_spanning_tree(G, weight='weight')
    
    # Find longest path
    longest_path = []
    max_length = 0
    
    # Try different start and end points
    nodes = list(mst.nodes())
    for start in nodes:
        for end in nodes:
            if start != end:
                try:
                    path = nx.shortest_path(mst, start, end, weight='weight')
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
                except nx.NetworkXNoPath:
                    continue
    
    if not longest_path:
        # Fallback: use any path
        longest_path = list(sections)
    
    print(f"MST with connectivity complete. Path length: {len(longest_path)}")
    return longest_path

def evaluate_order_quality(order: List[str], original_scores: Dict) -> Dict[str, float]:
    """Evaluate order quality using original SSIM scores."""
    if len(order) < 2:
        return {'total_score': 0, 'avg_score': 0, 'length': len(order)}
    
    # Load original SSIM scores for evaluation
    ssim_scores = {}
    with open('pair_ssim_ss10.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) >= 9:
                try:
                    sec1, sec2 = row[0].strip(), row[1].strip()
                    ssim_score = float(row[8])
                    ssim_scores[(sec1, sec2)] = ssim_score
                    ssim_scores[(sec2, sec1)] = ssim_score
                except (ValueError, IndexError):
                    continue
    
    consecutive_scores = []
    for i in range(len(order) - 1):
        score = ssim_scores.get((order[i], order[i+1]), 0)
        consecutive_scores.append(score)
    
    total_score = sum(consecutive_scores)
    avg_score = total_score / len(consecutive_scores)
    
    # Calculate consistency
    consistency_scores = []
    for i in range(1, len(order) - 1):
        prev_score = ssim_scores.get((order[i-1], order[i]), 0)
        next_score = ssim_scores.get((order[i], order[i+1]), 0)
        if prev_score > 0 and next_score > 0:
            consistency = min(prev_score, next_score) / max(prev_score, next_score)
            consistency_scores.append(consistency)
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    
    return {
        'total_score': total_score,
        'avg_score': avg_score,
        'length': len(order),
        'consistency': avg_consistency,
        'min_score': min(consecutive_scores) if consecutive_scores else 0,
        'max_score': max(consecutive_scores) if consecutive_scores else 0,
        'std_score': np.std(consecutive_scores) if consecutive_scores else 0
    }

def main():
    ap = argparse.ArgumentParser(description='Complete graph MST-based EM section ordering')
    ap.add_argument('pair_csv', help='CSV with pairwise scores')
    ap.add_argument('--method', choices=['complete', 'connectivity'], default='connectivity', 
                   help='MST method to use')
    ap.add_argument('--output', default='complete_mst_order.txt', help='Output order file')
    args = ap.parse_args()
    
    # Load data with distances
    scores, sections = load_pair_scores(Path(args.pair_csv))
    print(f"Loaded {len(sections)} sections with {len(scores)} pairwise distances")
    
    # Run MST ordering
    if args.method == 'complete':
        order = complete_mst_ordering(scores, sections)
    else:
        order = mst_with_connectivity(scores, sections)
    
    # Evaluate using original SSIM scores
    metrics = evaluate_order_quality(order, scores)
    
    print(f"Final order length: {len(order)}")
    print(f"Average consecutive pair SSIM: {metrics['avg_score']:.4f}")
    print(f"Consistency: {metrics['consistency']:.4f}")
    print(f"SSIM range: {metrics['min_score']:.4f} - {metrics['max_score']:.4f}")
    
    # Save order
    with open(args.output, 'w') as f:
        f.write(' '.join(order))
    print(f"Saved order to {args.output}")

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 