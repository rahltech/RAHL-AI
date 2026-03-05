"""Unit tests for RAHL pipeline"""

import unittest
import torch
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rahl.pipeline import RAHLPipeline
from src.rahl.utils import save_video, load_video_frames

class TestRAHLPipeline(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = RAHLPipeline()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.model)
        print("✅ Pipeline initialization test passed")
    
    def test_generate_basic(self):
        """Test basic generation"""
        video_data = self.pipeline.generate(
            prompt="a simple test",
            num_frames=8,
            num_inference_steps=2  # Minimal steps for testing
        )
        
        self.assertIn('frames', video_data)
        self.assertIn('fps', video_data)
        self.assertEqual(len(video_data['frames']), 8)
        print("✅ Basic generation test passed")
    
    def test_save_video(self):
        """Test video saving functionality"""
        # Generate dummy frames
        dummy_frames = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        
        output_path = os.path.join(self.temp_dir, "test_video.mp4")
        save_video(dummy_frames, output_path, fps=8)
        
        self.assertTrue(os.path.exists(output_path))
        print("✅ Video saving test passed")
    
    def test_load_video(self):
        """Test video loading functionality"""
        # First save a video
        dummy_frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        output_path = os.path.join(self.temp_dir, "test_load.mp4")
        save_video(dummy_frames, output_path)
        
        # Then load it
        loaded_frames = load_video_frames(output_path)
        
        self.assertEqual(len(loaded_frames), 5)
        print("✅ Video loading test passed")
    
    def test_device_handling(self):
        """Test device handling"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = RAHLPipeline(device=device)
        
        self.assertEqual(pipeline.device.type, device if device == "cpu" else "cuda")
        print("✅ Device handling test passed")

if __name__ == '__main__':
    import numpy as np
    unittest.main()
