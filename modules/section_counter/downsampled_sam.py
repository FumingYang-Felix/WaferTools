#!/usr/bin/env python3
"""
Enhanced SAM Automatic Detection with Downsampling Support

This script provides SAM-based mask detection with configurable downsampling
for faster processing of large wafer images (e.g., 5k×5k → 1k×1k → upscale results).

Usage:
    python automatic_mask_detection_downsampled.py --image_path wafer.png --downsample_factor 5 --output_csv masks.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import time
from typing import List, Tuple, Dict, Optional
import logging

# SAM imports
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # modules/section_counter/ → project root
    return os.path.abspath(os.path.join(here, '..', '..'))

def _resolve_checkpoint_path(model_key: str) -> str:
    """
    Resolve local SAM checkpoint path robustly.
    Tries multiple common names and locations (project root, ./cache/).
    """
    key = model_key.lower()
    if 'vit_h' in key:
        names = ['sam_vit_h_4b8939.pth', 'sam_vit_h.pth']
    elif 'vit_l' in key:
        names = ['sam_vit_l_0b3195.pth', 'sam_vit_l.pth']
    else:
        names = ['sam_vit_b_01ec64.pth', 'sam_vit_b.pth']

    search_dirs = [
        os.getcwd(),
        _project_root(),
        os.path.join(_project_root(), 'cache'),
    ]
    for d in search_dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    # Fallback: return first name in CWD (will error with clear message)
    return names[0]

class DownsampledSAMDetector:
    """SAM-based detection with downsampling support."""
    
    def __init__(self, model_key: str = "sam1_vit_b", downsample_factor: int = 1):
        self.downsample_factor = downsample_factor
        self.original_shape = None
        self.downsampled_shape = None
        
        if not SAM_AVAILABLE:
            raise ImportError("SAM not available. Install with: pip install segment-anything torch torchvision")
        
        # Load SAM model using model configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Loading SAM model: {model_key} on {device}")
        
        try:
            from model_config import get_model_for_pipeline, ModelManager
            manager = ModelManager(model_key)
            self.model_config = manager.config
            
            # Model-dependent initialization
            if self.model_config.model_type == "sam1":
                # SAM 1 approach
                sam_predictor = get_model_for_pipeline(model_key, device)
                sam = sam_predictor.model
                
                # Configure SAM 1 mask generator
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,           # 减少点采样密度，避免zig-zag
                    pred_iou_thresh=0.8,          # 提高IoU阈值，获得更高质量的mask
                    stability_score_thresh=0.9,    # 提高稳定性阈值
                    crop_n_layers=1,              # Multi-scale cropping
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=800,     # 增加最小区域面积，过滤小片段
                )
                self.predictor_type = "sam1"
                
            elif self.model_config.model_type == "sam2":
                # SAM 2 approach - use the predictor directly for automatic mask generation
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                
                sam2_predictor = get_model_for_pipeline(model_key, device)
                
                # Configure SAM 2 mask generator 
                self.mask_generator = SAM2AutomaticMaskGenerator(
                    model=sam2_predictor.model,
                    points_per_side=32,
                    pred_iou_thresh=0.8,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=500,
                )
                self.predictor_type = "sam2"
                
        except ImportError as e:
            # Fallback to hardcoded SAM 1 if model_config not available
            logger.warning(f"model_config not available ({e}), falling back to local SAM 1 weights")
            # Map model_key to appropriate registry key
            reg_key = 'vit_b'
            if 'vit_h' in model_key.lower():
                reg_key = 'vit_h'
            elif 'vit_l' in model_key.lower():
                reg_key = 'vit_l'
            ckpt_path = _resolve_checkpoint_path(model_key)
            sam = sam_model_registry[reg_key](checkpoint=ckpt_path)
            sam.to(device=device)
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=32,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.85,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=500,
            )
            self.predictor_type = "sam1_fallback"
        
    def downsample_image(self, image: np.ndarray) -> np.ndarray:
        """Downsample image by specified factor."""
        if self.downsample_factor == 1:
            return image
            
        self.original_shape = image.shape[:2]
        new_height = self.original_shape[0] // self.downsample_factor
        new_width = self.original_shape[1] // self.downsample_factor
        self.downsampled_shape = (new_height, new_width)
        
        logger.info(f"📉 Downsampling from {self.original_shape} to {self.downsampled_shape} (factor: {self.downsample_factor})")
        
        # Use INTER_AREA for downsampling (best quality)
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return downsampled
    
    def upscale_masks(self, masks: List[Dict]) -> List[Dict]:
        """Upscale masks back to original resolution with boundary smoothing."""
        if self.downsample_factor == 1:
            return masks
            
        logger.info(f"📈 Upscaling {len(masks)} masks from {self.downsampled_shape} to {self.original_shape}")
        
        upscaled_masks = []
        for mask_data in masks:
            # Get the downsampled mask
            mask = mask_data['segmentation']
            
            # Upscale the mask
            upscaled_mask = cv2.resize(
                mask.astype(np.uint8), 
                (self.original_shape[1], self.original_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Apply boundary smoothing
            smoothed_mask = self._smooth_boundary(upscaled_mask)
            
            # Find contours in smoothed mask
            contours, _ = cv2.findContours(
                smoothed_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Scale other properties
                upscaled_data = mask_data.copy()
                upscaled_data['segmentation'] = smoothed_mask
                upscaled_data['area'] = int(upscaled_data['area'] * (self.downsample_factor ** 2))
                
                # Scale bounding box
                bbox = upscaled_data['bbox']
                upscaled_data['bbox'] = [
                    bbox[0] * self.downsample_factor,  # x
                    bbox[1] * self.downsample_factor,  # y
                    bbox[2] * self.downsample_factor,  # width
                    bbox[3] * self.downsample_factor   # height
                ]
                
                # Calculate properties from smoothed contour
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Update with actual upscaled values
                upscaled_data['area'] = int(area)
                upscaled_data['contour'] = largest_contour.squeeze()
                
                upscaled_masks.append(upscaled_data)
        
        return upscaled_masks
    
    def _smooth_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Smooth the boundary of a mask to reduce zig-zag effects."""
        # Convert to uint8 for morphological operations
        mask_uint8 = mask.astype(np.uint8)
        
        # 1. Small morphological closing to fill small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        
        # 2. Small morphological opening to remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        # 3. Find contours and simplify them
        contours, _ = cv2.findContours(mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 4. Simplify contour to reduce zig-zag
        # Calculate epsilon based on contour perimeter
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = 0.005 * perimeter  # Adjust this value to control smoothing
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 5. Create new mask from simplified contour
        smoothed_mask = np.zeros_like(mask_uint8)
        cv2.fillPoly(smoothed_mask, [simplified_contour], 1)
        
        return smoothed_mask.astype(bool)
    
    def detect_masks(self, image_path: str, min_area: int = 1000, max_area: int = 50000) -> List[Dict]:
        """Detect masks with downsampling support."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        logger.info(f"📷 Loaded image: {image_rgb.shape}")
        
        # Downsample if needed
        downsampled_image = self.downsample_image(image_rgb)
        
        # Adjust min_area for downsampled resolution
        adjusted_min_area = min_area // (self.downsample_factor ** 2) if self.downsample_factor > 1 else min_area
        adjusted_max_area = max_area // (self.downsample_factor ** 2) if self.downsample_factor > 1 else max_area
        
        logger.info(f"🎯 Detecting masks using {self.predictor_type} on {'downsampled' if self.downsample_factor > 1 else 'original'} image")
        logger.info(f"📊 Area filters: {adjusted_min_area} - {adjusted_max_area} (adjusted for downsampling)")
        
        # Generate masks
        start_time = time.time()
        masks = self.mask_generator.generate(downsampled_image)
        detection_time = time.time() - start_time
        
        logger.info(f"⏱️ SAM detection completed in {detection_time:.2f} seconds")
        logger.info(f"🔍 Found {len(masks)} initial masks")
        
        # Filter by area
        filtered_masks = []
        for mask_data in masks:
            area = mask_data['area']
            if adjusted_min_area <= area <= adjusted_max_area:
                filtered_masks.append(mask_data)
        
        logger.info(f"✅ {len(filtered_masks)} masks after area filtering")
        
        # Upscale masks back to original resolution
        upscaled_masks = self.upscale_masks(filtered_masks)
        
        return upscaled_masks

def masks_to_csv(masks: List[Dict], output_path: str) -> pd.DataFrame:
    """Convert SAM masks to CSV format compatible with existing pipeline."""
    data = []
    
    for i, mask_data in enumerate(masks):
        # Get contour points
        if 'contour' in mask_data:
            contour_points = mask_data['contour']
        else:
            # Extract contour from segmentation mask
            mask = mask_data['segmentation']
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_points = largest_contour.squeeze()
            else:
                continue
        
        # Ensure contour_points is 2D
        if contour_points.ndim == 1:
            contour_points = contour_points.reshape(-1, 2)
        
        # Create mask name
        mask_name = f"SAM_mask_{i+1}"
        
        # Calculate properties
        area = mask_data.get('area', cv2.contourArea(contour_points))
        bbox = mask_data.get('bbox', cv2.boundingRect(contour_points))
        
        # Format coordinates as list of [x, y] pairs
        coordinates_list = contour_points.tolist()
        
        data.append({
            'Mask_Name': mask_name,
            'Coordinates': str(coordinates_list),  # Store as string representation
            'Area': area,
            'BoundingBox_X': bbox[0] if len(bbox) > 0 else 0,
            'BoundingBox_Y': bbox[1] if len(bbox) > 1 else 0,
            'BoundingBox_Width': bbox[2] if len(bbox) > 2 else 0,
            'BoundingBox_Height': bbox[3] if len(bbox) > 3 else 0,
            'Confidence': mask_data.get('stability_score', 1.0)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Saved {len(df)} masks to {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='SAM Automatic Detection with Downsampling')
    parser.add_argument('--image_path', required=True, help='Path to wafer image')
    parser.add_argument('--output_csv', required=True, help='Output CSV file')
    parser.add_argument('--downsample_factor', type=int, default=5, 
                       help='Downsampling factor (default: 5, i.e., 5k×5k → 1k×1k)')
    parser.add_argument('--model', default='sam1_vit_b', help='SAM model key (sam1_vit_b, sam2_tiny, sam2_small, sam2_base, etc.)')
    parser.add_argument('--min_area', type=int, default=1000, help='Minimum mask area')
    parser.add_argument('--max_area', type=int, default=50000, help='Maximum mask area')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--vis_output', default='sam_detection_downsampled.png', help='Visualization output')
    
    args = parser.parse_args()
    
    # Create detector
    detector = DownsampledSAMDetector(args.model, args.downsample_factor)
    
    # Detect masks
    masks = detector.detect_masks(args.image_path, args.min_area, args.max_area)
    
    # Save to CSV
    df = masks_to_csv(masks, args.output_csv)
    
    # Create visualization if requested
    if args.visualize:
        create_visualization(args.image_path, masks, args.vis_output)
    
    print(f"\n🎉 Detection complete!")
    print(f"📊 Found {len(masks)} masks")
    print(f"💾 Results saved to: {args.output_csv}")
    if args.visualize:
        print(f"🖼️ Visualization saved to: {args.vis_output}")

def create_visualization(image_path: str, masks: List[Dict], output_path: str):
    """Create visualization of detected masks."""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    overlay = image.copy()
    
    for i, mask_data in enumerate(masks):
        # Get contour
        if 'contour' in mask_data:
            contour = mask_data['contour']
        else:
            mask = mask_data['segmentation']
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
            else:
                continue
        
        # Draw contour
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
        
        # Draw label
        if contour.size > 0:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f"{i+1}", (cx-10, cy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.imwrite(output_path, overlay)

if __name__ == "__main__":
    main() 
