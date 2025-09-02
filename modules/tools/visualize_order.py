#!/usr/bin/env python3
"""
visualize_order.py - Visualize ordering results
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def load_pair_scores(csv_path: Path):
    """Load pairwise similarity CSV."""
    scores = {}
    sections = set()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) >= 8:
                try:
                    sec1, sec2 = row[0].strip(), row[1].strip()
                    num_inliers = float(row[7])
                    
                    scores[(sec1, sec2)] = num_inliers
                    scores[(sec2, sec1)] = num_inliers
                    sections.add(sec1)
                    sections.add(sec2)
                except (ValueError, IndexError):
                    continue
    
    return scores, sorted(list(sections))

def load_order(order_file: Path):
    """Load order from file."""
    with open(order_file, 'r') as f:
        content = f.read().strip()
        return content.split()

def create_similarity_matrix(order: list, scores: dict):
    """Create similarity matrix for the order."""
    n = len(order)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i, j] = scores.get((order[i], order[j]), 0)
    
    return matrix

def plot_order_analysis(order: list, scores: dict, title: str):
    """Plot analysis of the order."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Create similarity matrix
    matrix = create_similarity_matrix(order, scores)
    
    # Plot similarity matrix
    im = axes[0, 0].imshow(matrix, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'{title} - Similarity Matrix')
    axes[0, 0].set_xlabel('Section Index')
    axes[0, 0].set_ylabel('Section Index')
    plt.colorbar(im, ax=axes[0, 0])
    
    # Plot consecutive pair scores
    consecutive_scores = []
    for i in range(len(order) - 1):
        score = scores.get((order[i], order[i+1]), 0)
        consecutive_scores.append(score)
    
    axes[0, 1].plot(consecutive_scores, 'b-', linewidth=1)
    axes[0, 1].set_title('Consecutive Pair Scores')
    axes[0, 1].set_xlabel('Pair Index')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot histogram of consecutive scores
    axes[1, 0].hist(consecutive_scores, bins=30, alpha=0.7, color='blue')
    axes[1, 0].set_title('Distribution of Consecutive Scores')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot cumulative score
    cumulative = np.cumsum(consecutive_scores)
    axes[1, 1].plot(cumulative, 'r-', linewidth=2)
    axes[1, 1].set_title('Cumulative Score')
    axes[1, 1].set_xlabel('Section Index')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def detect_breaks(order: list, scores: dict, threshold_percentile=80):
    """Detect potential breaks in the order."""
    consecutive_scores = []
    for i in range(len(order) - 1):
        score = scores.get((order[i], order[i+1]), 0)
        consecutive_scores.append(score)
    
    # Find low score regions (potential breaks)
    threshold = np.percentile(consecutive_scores, threshold_percentile)
    breaks = []
    
    for i, score in enumerate(consecutive_scores):
        if score < threshold:
            breaks.append(i)
    
    # Group consecutive breaks
    if breaks:
        grouped_breaks = []
        current_group = [breaks[0]]
        
        for i in range(1, len(breaks)):
            if breaks[i] - breaks[i-1] <= 2:  # Within 2 positions
                current_group.append(breaks[i])
            else:
                grouped_breaks.append(current_group)
                current_group = [breaks[i]]
        
        grouped_breaks.append(current_group)
        
        # Find center of each break group
        break_positions = [int(np.mean(group)) for group in grouped_breaks]
        return break_positions, consecutive_scores
    
    return [], consecutive_scores

def main():
    ap = argparse.ArgumentParser(description='Visualize ordering results')
    ap.add_argument('pair_csv', help='CSV with pairwise scores')
    ap.add_argument('order_files', nargs='+', help='Order files to visualize')
    args = ap.parse_args()
    
    # Load scores
    scores, sections = load_pair_scores(Path(args.pair_csv))
    print(f"Loaded {len(sections)} sections with {len(scores)} pairwise scores")
    
    # Analyze each order
    for order_file in args.order_files:
        try:
            order = load_order(Path(order_file))
            title = Path(order_file).stem
            
            print(f"\nAnalyzing {title}:")
            print(f"Order length: {len(order)}")
            
            # Calculate statistics
            consecutive_scores = []
            for i in range(len(order) - 1):
                score = scores.get((order[i], order[i+1]), 0)
                consecutive_scores.append(score)
            
            avg_score = np.mean(consecutive_scores)
            total_score = np.sum(consecutive_scores)
            
            print(f"Average consecutive score: {avg_score:.2f}")
            print(f"Total score: {total_score:.2f}")
            
            # Detect breaks
            breaks, scores_list = detect_breaks(order, scores)
            if breaks:
                print(f"Detected {len(breaks)} potential break points at positions: {breaks}")
                print("These might indicate separate components or ordering issues")
            else:
                print("No significant breaks detected")
            
            # Create visualization
            fig = plot_order_analysis(order, scores, title)
            fig.savefig(f'{title}_analysis.png', dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {title}_analysis.png")
            
        except Exception as e:
            print(f"Error analyzing {order_file}: {e}")

if __name__ == '__main__':
    main() 
 
 
 
 
 
 
 
 
 
 
 