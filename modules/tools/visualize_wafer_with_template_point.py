import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alignment_utils import estimate_rigid, create_template_bounding_box

# Configuration
TEMPLATE_POINT = np.array([1100.0, 3100.0])  # Template point coordinates
TRANSFORMATION_METHOD = 'rigid'  # or 'affine'

def visualize_wafer_with_template_point(csv_path, image_path=None):
    """
    Visualize wafer sections with template point tracking and transformed bounding boxes
    """
    # Read resampled contours
    df = pd.read_csv(csv_path)
    
    # Create output directory
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = 'contour_resample_visuals'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process contours
    transformed_data = []
    for _, row in df.iterrows():
        try:
            contour = np.array(eval(row['contour']))
            
            # Create template bounding box
            template_box = create_template_bounding_box(contour)
            
            # Calculate transformation
            R, t = estimate_rigid(template_box, contour)
            
            # Transform template point
            transformed_point = R @ TEMPLATE_POINT + t
            
            # Transform bounding box
            transformed_box = (R @ template_box.T).T + t
            
            transformed_data.append({
                'id': row['id'],
                'template_point': transformed_point.tolist(),
                'bounding_box': transformed_box.tolist()
            })
        except Exception as e:
            print(f"Error processing {row['id']}: {e}")
    
    # Save transformed data
    output_file = os.path.join(output_dir, f"{base_name}_transformed_data.csv")
    pd.DataFrame(transformed_data).to_csv(output_file, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 12))
    
    # Plot original image if provided
    if image_path and os.path.exists(image_path):
        img = plt.imread(image_path)
        plt.imshow(img)
    
    # Plot contours and transformed elements
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            contour = np.array(eval(row['contour']))
            transformed = transformed_data[i]
            
            # Plot contour
            plt.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=1, alpha=0.5)
            
            # Plot transformed template point
            point = np.array(transformed['template_point'])
            plt.plot(point[0], point[1], 'c*', markersize=10)
            
            # Plot transformed bounding box
            box = np.array(transformed['bounding_box'])
            plt.plot(box[:, 0], box[:, 1], 'c-', linewidth=2)
            plt.plot(box[0, 0], box[0, 1], 'cs', markersize=8)  # Corner marker
        except Exception as e:
            print(f"Error visualizing {row['id']}: {e}")
    
    # Plot template point
    plt.plot(TEMPLATE_POINT[0], TEMPLATE_POINT[1], 'r*', markersize=15, label='Template Point')
    
    plt.title('Wafer Sections with Template Point Tracking')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    # Save visualization
    vis_file = os.path.join(output_dir, f"{base_name}_template_point_tracking.png")
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated visualization with template point tracking")
    print(f"Results saved to {output_file}")
    print(f"Visualization saved to {vis_file}")
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize_wafer_with_template_point.py <resampled_contours_csv> [image_path]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_wafer_with_template_point(csv_path, image_path) 