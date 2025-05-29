# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Comprehensive tests for SmolVLM Swin implementation

import unittest
import torch
import numpy as np
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.models.smolvlm.configuration_smolvlm_swin import (
    SmolVLMSwinConfig, 
    SmolVLMSwinVisionConfig
)
from transformers.models.smolvlm.modeling_smolvlm_swin import (
    SmolVLMSwinVisionTransformer,
    SmolVLMSwinModel,
    SmolVLMSwinForConditionalGeneration
)

@require_torch
class SmolVLMSwinModelTest(unittest.TestCase):
    """Test suite for SmolVLM Swin implementation"""
    
    def setUp(self):
        self.vision_config = SmolVLMSwinVisionConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
            use_swin_transformer=True,
            swin_window_size=7,
            num_swin_stages=4,
            swin_depths=[2, 2, 6, 2],
            swin_num_heads=[3, 6, 12, 24]
        )
        
        self.model_config = SmolVLMSwinConfig(vision_config=self.vision_config)
        
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config should not raise
        config = SmolVLMSwinVisionConfig()
        
        # Invalid depths length should raise
        with self.assertRaises(ValueError):
            SmolVLMSwinVisionConfig(
                num_swin_stages=4,
                swin_depths=[2, 2],  # Wrong length
                swin_num_heads=[3, 6, 12, 24]
            )
    
    def test_vision_transformer_forward(self):
        """Test vision transformer forward pass"""
        model = SmolVLMSwinVisionTransformer(self.vision_config)
        model.eval()
        
        # Test standard input
        batch_size, channels, height, width = 2, 3, 224, 224
        pixel_values = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            outputs = model(pixel_values)
        
        expected_seq_len = (height // self.vision_config.patch_size) ** 2
        self.assertEqual(outputs.last_hidden_state.shape, 
                        (batch_size, expected_seq_len, self.vision_config.hidden_size))
    
    def test_variable_resolution(self):
        """Test handling of variable resolution inputs"""
        model = SmolVLMSwinVisionTransformer(self.vision_config)
        model.eval()
        
        # Test different resolutions
        test_sizes = [(224, 224), (224, 448)]
        
        for h, w in test_sizes:
            with self.subTest(resolution=f"{h}x{w}"):
                pixel_values = torch.randn(1, 3, h, w)
                with torch.no_grad():
                    outputs = model(pixel_values)
                
                expected_patches = (h // self.vision_config.patch_size) * (w // self.vision_config.patch_size)
                self.assertEqual(outputs.last_hidden_state.shape[1], expected_patches)
    
    def test_attention_mask_handling(self):
        """Test proper attention mask handling"""
        model = SmolVLMSwinVisionTransformer(self.vision_config)
        model.eval()
        
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Create attention mask
        patch_size = self.vision_config.patch_size
        patch_attention_mask = torch.ones(
            (batch_size, 224 // patch_size, 224 // patch_size),
            dtype=torch.bool
        )
        
        with torch.no_grad():
            outputs = model(pixel_values, patch_attention_mask=patch_attention_mask)
        
        self.assertIsNotNone(outputs.last_hidden_state)
    
    def test_output_compatibility(self):
        """Test output compatibility with original SmolVLM"""
        # Swin model
        swin_model = SmolVLMSwinVisionTransformer(self.vision_config)
        
        # Standard model for comparison
        standard_config = SmolVLMSwinVisionConfig(**self.vision_config.to_dict())
        standard_config.use_swin_transformer = False
        standard_model = SmolVLMSwinVisionTransformer(standard_config)
        
        pixel_values = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            swin_outputs = swin_model(pixel_values)
            standard_outputs = standard_model(pixel_values)
        
        # Should have same output shapes
        self.assertEqual(swin_outputs.last_hidden_state.shape, 
                        standard_outputs.last_hidden_state.shape)
    
    def test_hierarchical_features(self):
        """Test hierarchical feature extraction"""
        model = SmolVLMSwinVisionTransformer(self.vision_config)
        model.eval()
        
        pixel_values = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=True)
        
        # Should have hidden states from all layers
        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.hidden_states), sum(self.vision_config.swin_depths) + 2)  # +2 for input and output
    
    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        model = SmolVLMSwinVisionTransformer(self.vision_config)
        model.train()
        
        pixel_values = torch.randn(1, 3, 224, 224, requires_grad=True)
        outputs = model(pixel_values)
        
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        
        # Check that gradients flow to input
        self.assertIsNotNone(pixel_values.grad)
        self.assertTrue(torch.any(pixel_values.grad != 0))
    
    def test_memory_efficiency(self):
        """Test memory efficiency compared to standard attention"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        device = torch.device("cuda")
        
        # Large input to test memory efficiency
        pixel_values = torch.randn(1, 3, 448, 448, device=device)
        
        # Swin model
        swin_model = SmolVLMSwinVisionTransformer(self.vision_config).to(device)
        swin_model.eval()
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            _ = swin_model(pixel_values)
            swin_memory = torch.cuda.max_memory_allocated()
        
        print(f"Swin Transformer memory usage: {swin_memory / 1024**2:.2f} MB")
    
    def test_full_model_integration(self):
        """Test full SmolVLM model with Swin transformer"""
        model = SmolVLMSwinForConditionalGeneration(self.model_config)
        model.eval()
        
        # Test image processing
        pixel_values = torch.randn(1, 1, 3, 224, 224)  # batch_size, num_images, channels, height, width
        input_ids = torch.randint(0, 1000, (1, 10))  # Simple token sequence
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values
            )
        
        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], 1)  # batch_size
        self.assertEqual(outputs.logits.shape[1], 10)  # sequence_length

class SmolVLMSwinBenchmarkTest(unittest.TestCase):
    """Benchmark tests for performance comparison"""
    
    @slow
    def test_inference_speed_comparison(self):
        """Compare inference speed between Swin and standard attention"""
        import time
        
        vision_config = SmolVLMSwinVisionConfig(
            hidden_size=768,
            num_hidden_layers=6,  # Smaller for faster testing
            image_size=224,
            patch_size=16
        )
        
        # Swin model
        swin_config = SmolVLMSwinVisionConfig(**vision_config.to_dict())
        swin_config.use_swin_transformer = True
        swin_model = SmolVLMSwinVisionTransformer(swin_config)
        swin_model.eval()
        
        # Standard model
        standard_config = SmolVLMSwinVisionConfig(**vision_config.to_dict())
        standard_config.use_swin_transformer = False
        standard_model = SmolVLMSwinVisionTransformer(standard_config)
        standard_model.eval()
        
        pixel_values = torch.randn(4, 3, 224, 224)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = swin_model(pixel_values)
                _ = standard_model(pixel_values)
        
        # Benchmark Swin
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = swin_model(pixel_values)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        swin_time = time.time() - start_time
        
        # Benchmark Standard
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = standard_model(pixel_values)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_time = time.time() - start_time
        
        print(f"Swin Transformer: {swin_time:.3f}s")
        print(f"Standard Attention: {standard_time:.3f}s")
        print(f"Speedup: {standard_time/swin_time:.2f}x")

if __name__ == "__main__":
    unittest.main()