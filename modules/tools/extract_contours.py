import os
import pandas as pd
import numpy as np
import ast
from skimage import measure

def extract_contours_from_csv(csv_path):
    """
    Extract contours from CSV file containing polygon coordinates
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = 'contour_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each row
    contours = []
    for _, row in df.iterrows():
        if row['id'].startswith(('Polygon', 'new_')):
            try:
                # Parse contour coordinates
                coords = ast.literal_eval(row['contour_coordinates'])
                if isinstance(coords[0][0], list):
                    coords = coords[0]
                contour = np.array(coords)
                contours.append({
                    'id': row['id'],
                    'contour': contour
                })
            except Exception as e:
                print(f"Error processing {row['id']}: {e}")
    
    # Save extracted contours
    output_file = os.path.join(output_dir, f"{base_name}_contours_extracted.csv")
    pd.DataFrame(contours).to_csv(output_file, index=False)
    
    print(f"Extracted {len(contours)} contours to {output_file}")
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_contours.py <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    extract_contours_from_csv(csv_path) 