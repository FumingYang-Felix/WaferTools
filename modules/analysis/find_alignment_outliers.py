import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alignment_utils import detect_alignment_outliers

def find_alignment_outliers(correlation_matrix_path):
    """
    Detect sections with poor alignment quality
    """
    # Read correlation matrix
    correlation_matrix = pd.read_csv(correlation_matrix_path).values
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(correlation_matrix_path))[0]
    output_dir = 'contour_resample_visuals'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Detect outliers
    outliers, outlier_scores, costs = detect_alignment_outliers(correlation_matrix)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot cost distribution
    plt.subplot(121)
    plt.hist(costs, bins=30, alpha=0.7)
    plt.axvline(np.percentile(costs, 95), color='r', linestyle='--', 
                label='95th percentile')
    plt.title('Alignment Cost Distribution')
    plt.xlabel('Cost')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot outlier scores
    plt.subplot(122)
    plt.bar(range(len(outliers)), outlier_scores)
    plt.title('Outlier Scores')
    plt.xlabel('Outlier Index')
    plt.ylabel('Score')
    
    plt.tight_layout()
    
    # Save visualization
    vis_file = os.path.join(output_dir, f"{base_name}_alignment_costs.png")
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save outlier information
    outlier_data = {
        'outlier_indices': outliers.tolist(),
        'outlier_scores': outlier_scores.tolist(),
        'costs': costs.tolist(),
        'threshold': float(np.percentile(costs, 95) * 2.0)
    }
    
    output_file = os.path.join(output_dir, f"{base_name}_outlier_info.csv")
    pd.DataFrame(outlier_data).to_csv(output_file, index=False)
    
    print(f"Found {len(outliers)} outliers")
    print(f"Results saved to {output_file}")
    print(f"Visualization saved to {vis_file}")
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python find_alignment_outliers.py <correlation_matrix_csv>")
        sys.exit(1)
    
    correlation_matrix_path = sys.argv[1]
    find_alignment_outliers(correlation_matrix_path) 