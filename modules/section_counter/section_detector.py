import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ipywidgets import interact, IntSlider
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils
from modules.section_counter.preprocess import preprocess_image
from modules.section_counter.filtering import filtering
import time
import cv2
import torch
from skimage.measure import find_contours
import logging

def automatic_identification(image_path, checkpoint, compress=False, apply_filtering=False, eps_values=None, min_samples_values=None, model_type='vit_h', force_recompute=False, **kwargs):
    start_time = time.time()
    print("\n[DEBUG] ===== automatic_identification called =====")
    print(f"[DEBUG] UI is using model_type: {model_type}")
    print(f"[DEBUG] imported checkpoint: {checkpoint}")
    # Default parameters
    default_params = {
        'points_per_side': 32,
        'pred_iou_thresh': 0.9,
        'stability_score_thresh': 0.95,
        'min_mask_region_area': 500,
        'output_mode': 'binary_mask',
        'device': 'cpu',
        'model_type': model_type
    }
    # Default filtering parameters
    if eps_values is None:
        eps_values = np.linspace(100, 1200, 11)
    if min_samples_values is None:
        min_samples_values = range(1, 5)
    params = {**default_params, **kwargs}
    print(f"[DEBUG] params: {params}")
    # Load and optionally preprocess the image
    if compress:
        print("[DEBUG] Compressing the image...")
        image = preprocess_image(image_path)
        print("[DEBUG] Image is compressed.")
    else:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        print(f"[DEBUG] Image shape: {image.shape}")
    # Cache file for storing generated masks
    file_directory = f"{os.path.splitext(image_path)[0]}_files"
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)
        print(f"[DEBUG] Created directory: {file_directory}")
    cache_tag = "compressed" if compress else "full"
    cache_file = f"{file_directory}/{os.path.basename(os.path.splitext(image_path)[0])}_{cache_tag}_masks.pkl"
    print(f"[DEBUG] cache_file: {cache_file}")
    if force_recompute and os.path.exists(cache_file):
        print(f"[DEBUG] force_recompute=True，auto delete cache file: {cache_file}")
        os.remove(cache_file)
    if os.path.exists(cache_file):
        print("[DEBUG] Cache file exists, loading cached results!")
        with open(cache_file, 'rb') as f:
            sorted_masks = pickle.load(f)
        print(f"[DEBUG] Loaded cached masks, number: {len(sorted_masks)}")
    else:
        print("[DEBUG] No cache, re-segmenting!")
        print(f"[DEBUG] SAM model initialization: model_type={params.get('model_type', 'vit_h')}, checkpoint={checkpoint}")
        sam = sam_model_registry[params.get('model_type', 'vit_h')](checkpoint)
        sam.to(device=params['device'])
        print(f"[DEBUG] SAM model initialized, device: {params['device']}")
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=params['points_per_side'],
            pred_iou_thresh=params['pred_iou_thresh'],
            stability_score_thresh=params['stability_score_thresh'],
            min_mask_region_area=params['min_mask_region_area'],
            output_mode=params['output_mode']
        )
        print("[DEBUG] Starting segmentation...")
        generated_masks = mask_generator.generate(image)
        print(f"[DEBUG] Segmentation completed, number of masks: {len(generated_masks)}")
        sorted_masks = sorted(generated_masks, key=lambda x: x['area'], reverse=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(sorted_masks, f)
        print("[DEBUG] Generated and cached masks.")
    if apply_filtering:
        print("[DEBUG] Filtering masks...")
        sorted_masks, chosen_params = filtering(sorted_masks, eps_values, min_samples_values)
        print(f"[DEBUG] Filtering completed with chosen parameters: {chosen_params}")
    print(f"[DEBUG] Returning {len(sorted_masks)} masks")
    print("[DEBUG] ===== automatic_identification end =====\n")
    print(f"[DEBUG] Total segmentation time: {time.time() - start_time:.2f} seconds")
    return sorted_masks

class SectionDetector:
    def __init__(self, model_type='vit_l', device=None):
        """
        Initialize the section detector with SAM model
        
        Args:
            model_type (str): Type of SAM model ('vit_h', 'vit_l', or 'vit_b')
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SAM model
        self.sam_model = None
        self.sam_predictor = None
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize SAM model"""
        try:
            # Get model checkpoint path
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', f'sam_{self.model_type}.pth')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
            
            # Initialize model
            self.sam_model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam_model.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            
            print(f"Initialized SAM model: {self.model_type} on {self.device}")
            
        except Exception as e:
            logging.error(f"Error initializing SAM model: {str(e)}")
            raise
    
    def process_image(self, image, downsample_ratio=1.0, patch_size=None):
        """
        Process image with SAM model
        
        Args:
            image (np.ndarray): Input image
            downsample_ratio (float): Downsample ratio for large images
            patch_size (int): Size of patches for processing large images
            
        Returns:
            list: List of detected masks with contours and scores
        """
        try:
            # Downsample if needed
            if downsample_ratio < 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * downsample_ratio), int(w * downsample_ratio)
                image = cv2.resize(image, (new_w, new_h))
            
            # Process in patches if needed
            if patch_size is not None:
                return self._process_patches(image, patch_size)
            
            # Set image
            self.sam_predictor.set_image(image)
            
            # Generate grid points
            h, w = image.shape[:2]
            points = self._generate_grid_points(h, w)
        
            # Process each point
            masks = []
            for point in points:
                # Convert point coordinates
                point_coords = np.array([[point[0], point[1]]])
                point_labels = np.array([1])
                
                # Predict mask
                mask, score, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
                
                if score[0] > 0.5:  # Only keep high confidence masks
                    # Find contours
                    contours = find_contours(mask[0], 0.5)
                    if contours:
                        # Use largest contour
                        contour = max(contours, key=len)
                        # Convert coordinates
                        contour = np.flip(contour, axis=1)
                        masks.append({
                            'contour': contour,
                            'score': float(score[0])
                        })
            
            return masks
            
        except Exception as e:
            logging.error(f"Error in process_image: {str(e)}")
            raise
    
    def _process_patches(self, image, patch_size):
        """Process large image in patches"""
        height, width = image.shape[:2]
        
        # Calculate patch grid
        n_patches_h = (height + patch_size - 1) // patch_size
        n_patches_w = (width + patch_size - 1) // patch_size
        
        all_masks = []
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch boundaries
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, height)
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, width)
                
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end]
                
                # Process patch
                patch_masks = self.process_image(patch)
                
                # Adjust mask coordinates
                for mask in patch_masks:
                    if mask is not None:
                        mask['contour'] = mask['contour'] + np.array([x_start, y_start])
                        all_masks.append(mask)
        
        return all_masks
    
    def _generate_grid_points(self, height, width, grid_size=32):
        """Generate grid points for SAM prediction"""
        # Calculate grid points
        x_points = np.linspace(0, width-1, grid_size)
        y_points = np.linspace(0, height-1, grid_size)
        
        # Create grid
        xx, yy = np.meshgrid(x_points, y_points)
        
        # Flatten and combine coordinates
        points = np.column_stack((xx.ravel(), yy.ravel()))
        
        return points
    
    def cleanup(self):
        """Clean up model resources"""
        if self.sam_model is not None:
            del self.sam_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.sam_model = None
            self.sam_predictor = None