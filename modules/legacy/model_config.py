"""
Model Configuration System for SAM Pipeline
Supports both SAM 1 and SAM 2 with various model sizes
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for SAM models"""
    name: str
    model_type: str  # 'sam1' or 'sam2'
    checkpoint_path: str
    parameters: int  # Number of parameters in millions
    file_size_mb: int
    description: str
    config_file: Optional[str] = None  # For SAM 2 models

# Available model configurations
MODEL_CONFIGS = {
    # SAM 1 Models
    "sam1_vit_b": ModelConfig(
        name="SAM 1 ViT-B",
        model_type="sam1",
        checkpoint_path="sam_vit_b.pth",
        parameters=91,
        file_size_mb=358,
        description="SAM 1 Base model - Good balance of speed and accuracy"
    ),
    "sam1_vit_l": ModelConfig(
        name="SAM 1 ViT-L", 
        model_type="sam1",
        checkpoint_path="sam_vit_l.pth",
        parameters=308,
        file_size_mb=1200,
        description="SAM 1 Large model - Better accuracy, slower"
    ),
    "sam1_vit_h": ModelConfig(
        name="SAM 1 ViT-H",
        model_type="sam1", 
        checkpoint_path="sam_vit_h.pth",
        parameters=636,
        file_size_mb=2400,
        description="SAM 1 Huge model - Best accuracy, slowest"
    ),
    
    # SAM 2 Models
    "sam2_tiny": ModelConfig(
        name="SAM 2 Hiera-Tiny",
        model_type="sam2",
        checkpoint_path="sam2_hiera_tiny.pt",
        parameters=38,
        file_size_mb=149,
        description="SAM 2 Tiny model - Fastest, good for real-time applications"
    ),
    "sam2_small": ModelConfig(
        name="SAM 2 Hiera-Small", 
        model_type="sam2",
        checkpoint_path="sam2_hiera_small.pt",
        parameters=46,
        file_size_mb=176,
        description="SAM 2 Small model - Fast with good accuracy"
    ),
    "sam2_base": ModelConfig(
        name="SAM 2 Hiera-Base",
        model_type="sam2",
        checkpoint_path="sam2_hiera_base_plus.pt", 
        parameters=80,
        file_size_mb=319,
        description="SAM 2 Base model - Recommended for most applications"
    ),
    "sam2_large": ModelConfig(
        name="SAM 2 Hiera-Large",
        model_type="sam2",
        checkpoint_path="sam2_hiera_large.pt",
        parameters=224,
        file_size_mb=896,
        description="SAM 2 Large model - Best accuracy for SAM 2"
    )
}

class ModelManager:
    """Manager class for SAM model loading and configuration"""
    
    def __init__(self, model_key: str = "sam1_vit_b"):
        """Initialize with a model configuration
        
        Args:
            model_key: Key from MODEL_CONFIGS dict
        """
        if model_key not in MODEL_CONFIGS:
            available = list(MODEL_CONFIGS.keys())
            raise ValueError(f"Model '{model_key}' not found. Available: {available}")
            
        self.config = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model = None
        
    def check_model_file(self) -> bool:
        """Check if model checkpoint file exists"""
        return os.path.exists(self.config.checkpoint_path)
        
    def get_download_url(self) -> str:
        """Get download URL for the model checkpoint"""
        urls = {
            "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
            "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
            "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        }
        return urls.get(self.config.checkpoint_path, "URL not found")
        
    def load_model(self, device: str = "cpu"):
        """Load the SAM model
        
        Args:
            device: Device to load model on ('cpu', 'cuda', 'mps')
            
        Returns:
            Loaded SAM model
        """
        if not self.check_model_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.config.checkpoint_path}\n"
                f"Download from: {self.get_download_url()}"
            )
            
        try:
            if self.config.model_type == "sam1":
                from segment_anything import sam_model_registry, SamPredictor
                model = sam_model_registry[self._get_sam1_model_type()](
                    checkpoint=self.config.checkpoint_path
                )
                model.to(device)
                return SamPredictor(model)
                
            elif self.config.model_type == "sam2": 
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                model = build_sam2(
                    config_file=self._get_sam2_config(),
                    ckpt_path=self.config.checkpoint_path,
                    device=device
                )
                return SAM2ImagePredictor(model)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.config.name}: {e}")
            
    def _get_sam1_model_type(self) -> str:
        """Get SAM 1 model type string"""
# model_config.py  →  ModelManager._get_sam1_model_type()
        mapping = {
            "sam_vit_b.pth": "vit_b",
            "sam_vit_b_01ec64.pth": "vit_b",
            "sam_vit_l.pth": "vit_l",          # ← 加这一行
            "sam_vit_l_0b3195.pth": "vit_l",
            "sam_vit_h.pth": "vit_h",          # ← 加这一行
            "sam_vit_h_4b8939.pth": "vit_h",
        }

        return mapping.get(self.config.checkpoint_path, "vit_b")
        
    def _get_sam2_config(self) -> str:
        """Get SAM 2 configuration file path"""
        # SAM 2 config files - use the correct paths found in the package
        config_mapping = {
            "sam2_hiera_tiny.pt": "sam2_hiera_t.yaml",
            "sam2_hiera_small.pt": "sam2_hiera_s.yaml", 
            "sam2_hiera_base_plus.pt": "sam2_hiera_b+.yaml",
            "sam2_hiera_large.pt": "sam2_hiera_l.yaml"
        }
        return config_mapping.get(self.config.checkpoint_path, "sam2_hiera_t.yaml")

def print_available_models():
    """Print information about all available models"""
    print("🔬 Available SAM Models:")
    print("=" * 80)
    
    sam1_models = {k: v for k, v in MODEL_CONFIGS.items() if v.model_type == "sam1"}
    sam2_models = {k: v for k, v in MODEL_CONFIGS.items() if v.model_type == "sam2"}
    
    print("\n📊 SAM 1 Models:")
    for key, config in sam1_models.items():
        status = "✅" if os.path.exists(config.checkpoint_path) else "❌"
        print(f"  {status} {key:<15} | {config.parameters:>3}M params | {config.file_size_mb:>4}MB | {config.description}")
        
    print("\n🚀 SAM 2 Models:")  
    for key, config in sam2_models.items():
        status = "✅" if os.path.exists(config.checkpoint_path) else "❌"
        print(f"  {status} {key:<15} | {config.parameters:>3}M params | {config.file_size_mb:>4}MB | {config.description}")
        
    print("\n💡 Usage Examples:")
    print("  python automatic_mask_detection.py --model sam1_vit_b")
    print("  python automatic_mask_detection.py --model sam2_tiny") 
    print("  python run_downsampled_sam_pipeline.py 42 15 10 --model sam2_small")

def get_model_for_pipeline(model_key: str = "sam1_vit_b", device: str = "cpu"):
    """Convenience function to get a model for pipeline use
    
    Args:
        model_key: Model configuration key
        device: Device to load on
        
    Returns:
        Loaded SAM predictor
    """
    manager = ModelManager(model_key)
    return manager.load_model(device)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        try:
            manager = ModelManager(model_key)
            print(f"✅ Model configuration loaded: {manager.config.name}")
            print(f"📁 Checkpoint: {manager.config.checkpoint_path}")
            print(f"📊 Parameters: {manager.config.parameters}M") 
            print(f"💾 File size: {manager.config.file_size_mb}MB")
            print(f"📝 Description: {manager.config.description}")
            
            if manager.check_model_file():
                print("✅ Model checkpoint file found")
            else:
                print("❌ Model checkpoint file not found")
                print(f"🔗 Download from: {manager.get_download_url()}")
                
        except ValueError as e:
            print(f"❌ Error: {e}")
    else:
        print_available_models() 
