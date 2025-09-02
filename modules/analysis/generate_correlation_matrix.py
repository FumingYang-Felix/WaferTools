import os
import pandas as pd
import numpy as np
from alignment_utils import calculate_signature, circular_cross_correlation
import matplotlib.pyplot as plt
from PIL import Image

def generate_correlation_matrix(csv_path):
    """
    Generate circular cross-correlation matrix for all sections
    """
    # Read resampled contours
    df = pd.read_csv(csv_path)
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = 'contour_resample_visuals'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate signatures for all contours
    signatures = []
    for _, row in df.iterrows():
        try:
            contour = np.array(eval(row['contour']))
            signature = calculate_signature(contour)
            signatures.append(signature)
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
    
    # Calculate correlation matrix
    n_contours = len(signatures)
    correlation_matrix = np.zeros((n_contours, n_contours))
    shift_matrix = np.zeros((n_contours, n_contours), dtype=int)
    
    for i in range(n_contours):
        for j in range(n_contours):
            if i != j:
                shift, score = circular_cross_correlation(signatures[i], signatures[j])
                correlation_matrix[i, j] = score
                shift_matrix[i, j] = shift
    
    # Save correlation matrix
    output_file = os.path.join(output_dir, f"{base_name}_circular_xcorr_matrix.csv")
    pd.DataFrame(correlation_matrix).to_csv(output_file, index=False)
    
    # Save shift matrix
    shift_file = os.path.join(output_dir, f"{base_name}_shift_matrix.csv")
    pd.DataFrame(shift_matrix).to_csv(shift_file, index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='viridis')
    plt.colorbar(label='Correlation Score')
    plt.title('Circular Cross-Correlation Matrix')
    
    # Save visualization
    vis_file = os.path.join(output_dir, f"{base_name}_correlation_matrix.png")
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated correlation matrix for {n_contours} contours")
    print(f"Results saved to {output_file} and {shift_file}")
    print(f"Visualization saved to {vis_file}")
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_correlation_matrix.py <resampled_contours_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    generate_correlation_matrix(csv_path) 