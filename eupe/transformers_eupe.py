# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the FAIR Noncommercial Research License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from eupe.models.vision_transformer import DinoVisionTransformer


class EupeViTConfig(PretrainedConfig):
    """Transformers configuration for EUPE ViT backbones."""

    model_type = "eupe_vit"

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: str = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = 2.0,
        pos_embed_rope_dtype: str = "fp32",
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = 1.0e-05,
        norm_layer: str = "layernormbf16",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.pos_embed_rope_base = pos_embed_rope_base
        self.pos_embed_rope_min_period = pos_embed_rope_min_period
        self.pos_embed_rope_max_period = pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords = pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords = pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords = pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords = pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ffn_ratio = ffn_ratio
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.layerscale_init = layerscale_init
        self.norm_layer = norm_layer
        self.ffn_layer = ffn_layer
        self.ffn_bias = ffn_bias
        self.proj_bias = proj_bias
        self.n_storage_tokens = n_storage_tokens
        self.mask_k_bias = mask_k_bias


@dataclass
class EupeViTPreset:
    """Named preset container for EUPE ViT configuration overrides."""

    name: str
    config: dict


EUPE_VIT_PRESETS = {
    "vitt16": EupeViTPreset(
        name="eupe-vitt16",
        config=dict(embed_dim=192, depth=12, num_heads=3),
    ),
    "vits16": EupeViTPreset(
        name="eupe-vits16",
        config=dict(embed_dim=384, depth=12, num_heads=6),
    ),
    "vitb16": EupeViTPreset(
        name="eupe-vitb16",
        config=dict(embed_dim=768, depth=12, num_heads=12),
    ),
}


class EupeViTModel(PreTrainedModel):
    """Transformers-compatible model wrapper around EUPE DinoVisionTransformer."""

    config_class = EupeViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    _no_split_modules = ["SelfAttentionBlock"]

    def __init__(self, config: EupeViTConfig):
        super().__init__(config)
        self.vit = DinoVisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            pos_embed_rope_base=config.pos_embed_rope_base,
            pos_embed_rope_min_period=config.pos_embed_rope_min_period,
            pos_embed_rope_max_period=config.pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=config.pos_embed_rope_normalize_coords,
            pos_embed_rope_shift_coords=config.pos_embed_rope_shift_coords,
            pos_embed_rope_jitter_coords=config.pos_embed_rope_jitter_coords,
            pos_embed_rope_rescale_coords=config.pos_embed_rope_rescale_coords,
            pos_embed_rope_dtype=config.pos_embed_rope_dtype,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            ffn_ratio=config.ffn_ratio,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            layerscale_init=config.layerscale_init,
            norm_layer=config.norm_layer,
            ffn_layer=config.ffn_layer,
            ffn_bias=config.ffn_bias,
            proj_bias=config.proj_bias,
            n_storage_tokens=config.n_storage_tokens,
            mask_k_bias=config.mask_k_bias,
        )
        self.vit.init_weights()

    def _init_weights(self, module: nn.Module) -> None:
        # Signature required by PreTrainedModel; initialization is delegated to DinoVisionTransformer.
        del module
        pass

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        feats = self.vit.forward_features(pixel_values)
        cls_token = feats["x_norm_clstoken"].unsqueeze(1)
        patch_tokens = feats["x_norm_patchtokens"]
        last_hidden_state = torch.cat([cls_token, patch_tokens], dim=1)
        pooler_output = feats["x_norm_clstoken"]

        if not return_dict:
            return (last_hidden_state, pooler_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )


def register_eupe_transformers() -> None:
    """Register EUPE config/model for AutoConfig and AutoModel."""
    AutoConfig.register(EupeViTConfig.model_type, EupeViTConfig)
    AutoModel.register(EupeViTConfig, EupeViTModel)
