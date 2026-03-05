"""Unit tests for RAHL model"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rahl.model import RAHLModel, TemporalModule

class TestRAHLModel(unittest.TestCase):
    
    def setUp(self):
        self.model = RAHLModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.vae)
        self.assertIsNotNone(self.model.unet)
        self.assertIsNotNone(self.model.text_encoder)
        print("✅ Model initialization test passed")
    
    def test_temporal_module(self):
        """Test temporal module forward pass"""
        temp_mod = TemporalModule(dim=320, num_frames=16)
        batch = 1
        frames = 16
        channels = 320
        height = 32
        width = 32
        
        x = torch.randn(batch, frames, channels, height, width)
        output = temp_mod(x)
        
        self.assertEqual(output.shape, x.shape)
        print("✅ Temporal module test passed")
    
    def test_text_encoding(self):
        """Test text encoding"""
        prompt = "a test prompt"
        text_input = self.model.tokenizer(prompt, padding="max_length", 
                                         max_length=77, return_tensors="pt")
        text_embeddings = self.model.text_encoder(text_input.input_ids)[0]
        
        self.assertEqual(text_embeddings.shape[0], 1)
        self.assertEqual(text_embeddings.shape[1], 77)
        print("✅ Text encoding test passed")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory(self):
        """Test GPU memory usage"""
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            _ = self.model.to("cuda")
            final_memory = torch.cuda.memory_allocated()
            print(f"✅ GPU memory test passed - Memory change: {final_memory - initial_memory} bytes")

if __name__ == '__main__':
    unittest.main()
