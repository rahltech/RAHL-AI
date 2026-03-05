"""Configuration settings for RAHL"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class RAHLConfig:
    """Configuration class for RAHL model"""
    
    # Model settings
    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_frames: int = 16
    vae_scale_factor: int = 8
    
    # Generation settings
    default_height: int = 512
    default_width: int = 512
    default_fps: int = 8
    default_guidance_scale: float = 7.5
    default_num_inference_steps: int = 50
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    enable_half_precision: bool = torch.cuda.is_available()
    
    # Memory optimization
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_model_cpu_offload: bool = False
    
    # Output settings
    output_format: str = "mp4"  # mp4, gif, frames
    save_intermediate: bool = False
    
    def __post_init__(self):
        """Validate configuration"""
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        
        if self.default_height % 64 != 0 or self.default_width % 64 != 0:
            print("⚠️ Height and width should be multiples of 64 for best results")

# Global configuration instance
config = RAHLConfig()
