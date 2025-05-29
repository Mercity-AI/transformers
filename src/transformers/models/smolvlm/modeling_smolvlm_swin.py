# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# SmolVLM with Swin Transformer vision encoder

import math
import warnings
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging, ModelOutput
from .configuration_smolvlm_swin import SmolVLMSwinConfig, SmolVLMSwinVisionConfig
from .modeling_smolvlm import (
    SmolVLMPreTrainedModel, 
    SmolVLMVisionEmbeddings, 
    SmolVLMVisionMLP,
    SmolVLMModel,
    SmolVLMForConditionalGeneration,
    SmolVLMBaseModelOutputWithPast,
    SmolVLMCausalLMOutputWithPast
)

logger = logging.get_logger(__name__)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def window_partition(x, window_size):
    """
    Partition input tensor into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partitioning back to original tensor.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.fused_window_process = fused_window_process
        
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # Partition windows
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # Partition windows
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SmolVLMSwinEncoder(nn.Module):
    """Swin Transformer backbone for SmolVLM"""

    def __init__(self, config: SmolVLMSwinVisionConfig):
        super().__init__()
        self.config = config
        
        # Hierarchical parameters
        self.num_layers = len(config.swin_depths)
        self.embed_dim = config.hidden_size
        self.patch_norm = True
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = 4.
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, config.swin_drop_path_rate, sum(config.swin_depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Calculate layer parameters
            layer_dim = int(self.embed_dim * 2 ** i_layer)
            input_resolution = (
                config.image_size // config.patch_size // (2 ** i_layer),
                config.image_size // config.patch_size // (2 ** i_layer)
            )
            
            layer = BasicLayer(
                dim=layer_dim,
                input_resolution=input_resolution,
                depth=config.swin_depths[i_layer],
                num_heads=config.swin_num_heads[i_layer],
                window_size=config.swin_window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=config.swin_qkv_bias,
                qk_scale=config.swin_qk_scale,
                drop=0.,
                attn_drop=config.attention_dropout,
                drop_path=dpr[sum(config.swin_depths[:i_layer]):sum(config.swin_depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=config.use_checkpoint,
                fused_window_process=config.fused_window_process
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        
        # Feature adaptation for compatibility with original SmolVLM
        if config.maintain_resolution_compatibility:
            self.feature_adapter = nn.Sequential(
                nn.Linear(self.num_features, config.hidden_size),
                nn.LayerNorm(config.hidden_size)
            )
            
            # Spatial upsampling to maintain patch count
            original_patches = (config.image_size // config.patch_size) ** 2
            final_patches = (config.image_size // config.patch_size // (2 ** (self.num_layers - 1))) ** 2
            
            if final_patches != original_patches:
                self.spatial_adapter = nn.ConvTranspose2d(
                    config.hidden_size, config.hidden_size,
                    kernel_size=2**(self.num_layers-1), stride=2**(self.num_layers-1)
                )
        else:
            self.feature_adapter = None
            self.spatial_adapter = None

        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        """Forward pass maintaining compatibility with SmolVLM interface"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        x = inputs_embeds
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Pass through Swin layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        x = self.norm(x)
        
        # Adapt features back to original format if needed
        if self.feature_adapter is not None:
            # Adapt feature dimensions
            x = self.feature_adapter(x)
            
            # Adapt spatial dimensions if needed
            if self.spatial_adapter is not None:
                B, L, C = x.shape
                H = W = int(math.sqrt(L))
                x = x.transpose(1, 2).view(B, C, H, W)
                x = self.spatial_adapter(x)
                x = x.flatten(2).transpose(1, 2)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if not return_dict:
            return tuple(v for v in [x, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

class SmolVLMSwinVisionTransformer(SmolVLMPreTrainedModel):
    """SmolVLM Vision Transformer with Swin backbone"""
    
    config_class = SmolVLMSwinVisionConfig
    
    def __init__(self, config: SmolVLMSwinVisionConfig):
        super().__init__(config)
        
        self.embeddings = SmolVLMVisionEmbeddings(config)
        
        # Use Swin encoder if enabled, otherwise fall back to standard
        if config.use_swin_transformer:
            self.encoder = SmolVLMSwinEncoder(config)
        else:
            from .modeling_smolvlm import SmolVLMEncoder
            self.encoder = SmolVLMEncoder(config)
        
        self.patch_size = config.patch_size
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """Forward pass maintaining full compatibility with original SmolVLM"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.patch_size
            patch_attention_mask = torch.ones(
                (batch_size, pixel_values.size(2) // patch_size, pixel_values.size(3) // patch_size)
            ).to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        
        # Handle attention mask for different attention implementations
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif not self._use_flash_attention_2:
            from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class SmolVLMSwinModel(SmolVLMModel):
    """Enhanced SmolVLM Model with Swin Transformer vision encoder"""
    
    config_class = SmolVLMSwinConfig
    
    def __init__(self, config: SmolVLMSwinConfig):
        # Initialize parent but replace vision model
        super().__init__(config)
        self.vision_model = SmolVLMSwinVisionTransformer._from_config(config.vision_config)

class SmolVLMSwinForConditionalGeneration(SmolVLMForConditionalGeneration):
    """SmolVLM for Conditional Generation with Swin Transformer vision encoder"""
    
    config_class = SmolVLMSwinConfig
    
    def __init__(self, config: SmolVLMSwinConfig):
        super().__init__(config)
        self.model = SmolVLMSwinModel(config)