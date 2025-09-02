import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alignment_utils import resample_contour

def resample_and_visualize_contours(csv_path, n_points=100):
    """
    Resample contours to uniform number of points and visualize
    """
    # Read extracted contours
    df = pd.read_csv(csv_path)
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = 'contour_resample_visuals'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each contour
    resampled_contours = []
    for _, row in df.iterrows():
        try:
            # Parse contour
            contour = np.array(eval(row['contour']))
            
            # Resample contour
            resampled = resample_contour(contour, n_points)
            
            resampled_contours.append({
                'id': row['id'],
                'contour': resampled.tolist()
            })
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
    
    # Save resampled contours
    output_file = os.path.join(output_dir, f"{base_name}_contours_resampled.csv")
    pd.DataFrame(resampled_contours).to_csv(output_file, index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    for contour_data in resampled_contours:
        contour = np.array(contour_data['contour'])
        plt.plot(contour[:, 0], contour[:, 1], '-', linewidth=1)
    
    plt.title('Resampled Contours')
    plt.axis('equal')
    plt.grid(True)
    
    # Save visualization
    vis_file = os.path.join(output_dir, f"{base_name}_resampled_visualization.png")
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Resampled {len(resampled_contours)} contours to {output_file}")
    print(f"Visualization saved to {vis_file}")
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python resample_and_visualize_contours.py <contours_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    resample_and_visualize_contours(csv_path) 