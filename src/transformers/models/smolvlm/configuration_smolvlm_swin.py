# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Enhanced SmolVLM configuration with Swin Transformer support

from typing import List, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from .configuration_smolvlm import SmolVLMVisionConfig, SmolVLMConfig

logger = logging.get_logger(__name__)

class SmolVLMSwinVisionConfig(SmolVLMVisionConfig):
    """
    Enhanced SmolVLM Vision Configuration with Swin Transformer support.
    
    This configuration class stores the configuration of a SmolVLM vision model with 
    hierarchical Swin Transformer blocks for improved efficiency and feature learning.
    """
    
    model_type = "smolvlm_swin_vision"
    
    def __init__(
        self,
        # Swin Transformer specific parameters
        use_swin_transformer: bool = True,
        swin_window_size: int = 7,
        swin_shift_size: Optional[int] = None,
        
        # Hierarchical architecture parameters
        use_hierarchical_stages: bool = True,
        num_swin_stages: int = 4,
        swin_depths: List[int] = None,
        swin_num_heads: List[int] = None,
        
        # Training and regularization
        swin_drop_path_rate: float = 0.1,
        swin_qkv_bias: bool = True,
        swin_qk_scale: Optional[float] = None,
        
        # Feature processing
        hierarchical_feature_fusion: bool = True,
        maintain_resolution_compatibility: bool = True,
        
        # Performance optimizations
        use_checkpoint: bool = False,
        fused_window_process: bool = False,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Swin Transformer configuration
        self.use_swin_transformer = use_swin_transformer
        self.swin_window_size = swin_window_size
        self.swin_shift_size = swin_shift_size or (swin_window_size // 2)
        
        # Hierarchical stages configuration
        self.use_hierarchical_stages = use_hierarchical_stages
        self.num_swin_stages = num_swin_stages
        
        # Default depths and heads for 4-stage hierarchy
        if swin_depths is None:
            self.swin_depths = [2, 2, 6, 2]
        else:
            self.swin_depths = swin_depths
            
        if swin_num_heads is None:
            # Calculate heads based on hidden_size and stages
            base_heads = self.num_attention_heads
            self.swin_num_heads = [
                max(1, base_heads // (2**(3-i))) for i in range(num_swin_stages)
            ]
        else:
            self.swin_num_heads = swin_num_heads
        
        # Validate configuration
        self._validate_swin_config()
        
        # Training parameters
        self.swin_drop_path_rate = swin_drop_path_rate
        self.swin_qkv_bias = swin_qkv_bias
        self.swin_qk_scale = swin_qk_scale
        
        # Feature fusion
        self.hierarchical_feature_fusion = hierarchical_feature_fusion
        self.maintain_resolution_compatibility = maintain_resolution_compatibility
        
        # Performance
        self.use_checkpoint = use_checkpoint
        self.fused_window_process = fused_window_process
    
    def _validate_swin_config(self):
        """Validate Swin Transformer configuration parameters"""
        if self.use_hierarchical_stages:
            if len(self.swin_depths) != self.num_swin_stages:
                raise ValueError(
                    f"Length of swin_depths ({len(self.swin_depths)}) must equal "
                    f"num_swin_stages ({self.num_swin_stages})"
                )
            
            if len(self.swin_num_heads) != self.num_swin_stages:
                raise ValueError(
                    f"Length of swin_num_heads ({len(self.swin_num_heads)}) must equal "
                    f"num_swin_stages ({self.num_swin_stages})"
                )
        
        # Validate window size
        patch_grid_size = self.image_size // self.patch_size
        if patch_grid_size % self.swin_window_size != 0:
            logger.warning(
                f"Patch grid size ({patch_grid_size}) is not divisible by window size "
                f"({self.swin_window_size}). This may cause issues with window attention."
            )

class SmolVLMSwinConfig(SmolVLMConfig):
    """
    Enhanced SmolVLM Configuration with Swin Transformer vision encoder.
    """
    
    model_type = "smolvlm_swin"
    sub_configs = {"text_config": SmolVLMConfig.sub_configs["text_config"], 
                   "vision_config": SmolVLMSwinVisionConfig}
    
    def __init__(
        self,
        vision_config: Optional[Union[SmolVLMSwinVisionConfig, dict]] = None,
        **kwargs
    ):
        # Initialize with Swin vision config
        if vision_config is None:
            vision_config = SmolVLMSwinVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = SmolVLMSwinVisionConfig(**vision_config)
        
        super().__init__(vision_config=vision_config, **kwargs)