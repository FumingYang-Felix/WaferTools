#!/usr/bin/env python3
"""
inliers_order_builder.py - EM section ordering using num_inliers
===============================================================
Uses num_inliers as primary metric for ordering EM sections
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import csv

def load_pair_scores(csv_path: Path) -> Tuple[Dict[Tuple[str,str], float], List[str]]:
    """Load pairwise similarity CSV using num_inliers as primary metric."""
    scores = {}
    sections = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 8:
                try:
                    sec1, sec2 = row[0].strip(), row[1].strip()
                    num_inliers = float(row[7])  # num_inliers is in column 8 (index 7)
                    
                    # Use num_inliers directly as similarity score
                    # Higher num_inliers = higher similarity
                    scores[(sec1, sec2)] = num_inliers
                    scores[(sec2, sec1)] = num_inliers
                    sections.add(sec1)
                    sections.add(sec2)
                except (ValueError, IndexError):
                    continue
    
    return scores, sorted(list(sections))

def inliers_greedy_ordering(scores: Dict[Tuple[str,str], float], sections: List[str]) -> List[str]:
    """Greedy ordering using num_inliers."""
    print("Running inliers-based greedy ordering...")
    
    # Find the pair with highest num_inliers
    best_pair = None
    best_inliers = -1
    
    for (sec1, sec2), inliers in scores.items():
        if inliers > best_inliers:
            best_inliers = inliers
            best_pair = (sec1, sec2)
    
    if not best_pair:
        return sections
    
    # Start chain with best pair
    chain = list(best_pair)
    remaining = set(sections) - set(chain)
    
    # Grow chain greedily
    while remaining:
        best_extension = None
        best_inliers = -1
        add_to_front = False
        
        # Try extending from both ends
        for end_section in [chain[0], chain[-1]]:
            for neighbor in remaining:
                inliers = scores.get((end_section, neighbor), 0)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_extension = neighbor
                    add_to_front = (end_section == chain[0])
        
        if best_extension:
            if add_to_front:
                chain.insert(0, best_extension)
            else:
                chain.append(best_extension)
            remaining.remove(best_extension)
        else:
            # Add any remaining section
            chain.extend(list(remaining))
            break
    
    print(f"Inliers greedy ordering complete. Chain length: {len(chain)}")
    return chain

def multi_start_inliers(scores: Dict[Tuple[str,str], float], sections: List[str]) -> List[str]:
    """Multi-start greedy with num_inliers."""
    print("Running multi-start inliers ordering...")
    
    # Calculate average inliers for each section
    avg_inliers = {}
    for section in sections:
        inliers_list = [scores.get((section, other), 0) for other in sections if other != section]
        avg_inliers[section] = np.mean(inliers_list) if inliers_list else 0
    
    # Try starting from top 5 sections with highest average inliers
    top_starters = sorted(avg_inliers.items(), key=lambda x: x[1], reverse=True)[:5]
    
    best_chain = None
    best_score = -1
    
    for start_section, _ in top_starters:
        # Build chain starting from this section
        chain = [start_section]
        remaining = set(sections) - {start_section}
        
        while remaining:
            best_extension = None
            best_inliers = -1
            
            # Try extending from the end
            current = chain[-1]
            for neighbor in remaining:
                inliers = scores.get((current, neighbor), 0)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_extension = neighbor
            
            if best_extension:
                chain.append(best_extension)
                remaining.remove(best_extension)
            else:
                # Add any remaining section
                chain.extend(list(remaining))
                break
        
        # Evaluate this chain
        chain_score = evaluate_chain_inliers(chain, scores)
        if chain_score > best_score:
            best_score = chain_score
            best_chain = chain
    
    print(f"Multi-start inliers complete. Best average inliers: {best_score:.2f}")
    return best_chain

def evaluate_chain_inliers(chain: List[str], scores: Dict) -> float:
    """Evaluate chain quality using num_inliers."""
    if len(chain) < 2:
        return 0
    
    total_inliers = 0
    for i in range(len(chain) - 1):
        inliers = scores.get((chain[i], chain[i+1]), 0)
        total_inliers += inliers
    
    return total_inliers / (len(chain) - 1)

def evaluate_order_quality_inliers(order: List[str], scores: Dict) -> Dict[str, float]:
    """Evaluate order quality using num_inliers."""
    if len(order) < 2:
        return {'total_inliers': 0, 'avg_inliers': 0, 'length': len(order)}
    
    consecutive_inliers = []
    for i in range(len(order) - 1):
        inliers = scores.get((order[i], order[i+1]), 0)
        consecutive_inliers.append(inliers)
    
    total_inliers = sum(consecutive_inliers)
    avg_inliers = total_inliers / len(consecutive_inliers)
    
    # Calculate consistency
    consistency_scores = []
    for i in range(1, len(order) - 1):
        prev_inliers = scores.get((order[i-1], order[i]), 0)
        next_inliers = scores.get((order[i], order[i+1]), 0)
        if prev_inliers > 0 and next_inliers > 0:
            consistency = min(prev_inliers, next_inliers) / max(prev_inliers, next_inliers)
            consistency_scores.append(consistency)
    
    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    
    return {
        'total_inliers': total_inliers,
        'avg_inliers': avg_inliers,
        'length': len(order),
        'consistency': avg_consistency,
        'min_inliers': min(consecutive_inliers) if consecutive_inliers else 0,
        'max_inliers': max(consecutive_inliers) if consecutive_inliers else 0,
        'std_inliers': np.std(consecutive_inliers) if consecutive_inliers else 0
    }

def compare_with_ssim(order: List[str]) -> Dict[str, float]:
    """Compare inliers-based order with SSIM scores."""
    # Load SSIM scores for comparison
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
    
    consecutive_ssim = []
    for i in range(len(order) - 1):
        ssim = ssim_scores.get((order[i], order[i+1]), 0)
        consecutive_ssim.append(ssim)
    
    avg_ssim = np.mean(consecutive_ssim) if consecutive_ssim else 0
    
    return {
        'avg_ssim': avg_ssim,
        'total_ssim': sum(consecutive_ssim),
        'min_ssim': min(consecutive_ssim) if consecutive_ssim else 0,
        'max_ssim': max(consecutive_ssim) if consecutive_ssim else 0
    }

def main():
    ap = argparse.ArgumentParser(description='EM section ordering using num_inliers')
    ap.add_argument('pair_csv', help='CSV with pairwise scores')
    ap.add_argument('--method', choices=['greedy', 'multi'], default='multi', 
                   help='Ordering method to use')
    ap.add_argument('--output', default='inliers_order.txt', help='Output order file')
    args = ap.parse_args()
    
    # Load data with num_inliers
    scores, sections = load_pair_scores(Path(args.pair_csv))
    print(f"Loaded {len(sections)} sections with {len(scores)} pairwise inliers scores")
    
    # Run inliers-based ordering
    if args.method == 'greedy':
        order = inliers_greedy_ordering(scores, sections)
    else:
        order = multi_start_inliers(scores, sections)
    
    # Evaluate using inliers
    inliers_metrics = evaluate_order_quality_inliers(order, scores)
    
    # Compare with SSIM
    ssim_metrics = compare_with_ssim(order)
    
    print(f"Final order length: {len(order)}")
    print(f"Average consecutive pair inliers: {inliers_metrics['avg_inliers']:.2f}")
    print(f"Inliers consistency: {inliers_metrics['consistency']:.4f}")
    print(f"Inliers range: {inliers_metrics['min_inliers']:.0f} - {inliers_metrics['max_inliers']:.0f}")
    print(f"Corresponding average SSIM: {ssim_metrics['avg_ssim']:.4f}")
    
    # Save order
    with open(args.output, 'w') as f:
        f.write(' '.join(order))
    print(f"Saved order to {args.output}")

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 