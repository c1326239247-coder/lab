# -*- coding: utf-8 -*-
"""Extract a hierarchical 4-layer KV cache from a CLIP vision encoder.

We hook the four sub-modules of the manual's hierarchy to four chosen
layers of the CLIP encoder:

  basic              <- an early layer (rich texture / edge info)
  semantic_object    <- a mid layer (object-level features)
  spatial_relation   <- a late layer (relational features)
  attribute          <- the final layer (global attribute features)

This is a design choice exposed in the config.  Other assignments are
possible (e.g. four consecutive layers) -- the rest of the pipeline
does not depend on which exact layers we pick.

Usage
-----
    from cache_bridge_torch.extract import CLIPHierarchicalCacheExtractor
    from transformers import CLIPVisionModel

    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    extractor = CLIPHierarchicalCacheExtractor(cfg, clip)
    cache, coords = extractor(pixel_values)

`cache` is a dict of {layer_name: {"K": Tensor, "V": Tensor}}
with shape (B, vision_num_heads, T, vision_head_dim) per layer.
`coords` is a dict of {layer_name: Tensor} with shape (B, T, 2),
the (row, col) of each patch on the 2D grid.

The trick we use is to install forward hooks on the four chosen
encoder layers that record the (K, V) projections inside the
self-attention block.  This is the only way to access the
intermediate KV cache without re-implementing CLIP.
"""

import torch
import torch.nn as nn


class CLIPHierarchicalCacheExtractor(nn.Module):
    """Hooks into a CLIP vision encoder and produces the 4-layer cache."""

    def __init__(self, cfg, clip_vision_model):
        super().__init__()
        self.cfg = cfg
        self.clip = clip_vision_model
        # Pre-compute the (row, col) of every patch on the grid.
        n = cfg.image_size // cfg.patch_size  # 16 for ViT-L/14 on 224
        rows = torch.arange(n).repeat_interleave(n)
        cols = torch.arange(n).repeat(n)
        # (n*n, 2)
        self.register_buffer("patch_grid", torch.stack([rows, cols], dim=1).float())
        # Sanity: the CLIP position embedding should have n*n + 1 entries
        # (one for CLS, n*n for patches).
        n_patches = (cfg.image_size // cfg.patch_size) ** 2
        n_positions = self.clip.embeddings.position_embedding.weight.shape[0]
        assert n_positions == n_patches + 1, (
            "Expected %d positions (1 CLS + %d patches), got %d" %
            (n_patches + 1, n_patches, n_positions)
        )
        # Map of layer-name -> index in the encoder layers
        self.layer_indices = {
            "basic": cfg.basic_layer_idx,
            "semantic_object": cfg.object_layer_idx,
            "spatial_relation": cfg.relation_layer_idx,
            "attribute": cfg.attribute_layer_idx,
        }
        # Hooks we have installed
        self._handles = []
        self._captures = {}

    def _make_hook(self, name):
        def hook(module, args, kwargs):
            # We hook the *self-attention* forward so we can record
            # the (K, V) tensors that the attention block actually
            # uses.  module: CLIPEncoderLayer.self_attn
            hidden = args[0] if args else kwargs.get("hidden_states")
            B, T, D = hidden.shape
            # We replicate CLIP's self-attention QKV projection so we
            # can capture the per-head K, V.
            num_heads = self.cfg.vision_num_heads
            head_dim = self.cfg.vision_head_dim
            q = module.q_proj(hidden)
            k = module.k_proj(hidden)
            v = module.v_proj(hidden)
            k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, T, num_heads, head_dim).transpose(1, 2)
            self._captures[name] = {"K": k, "V": v}
        return hook

    def __enter__(self):
        for name, idx in self.layer_indices.items():
            layer = self.clip.encoder.layers[idx]
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_hook(name), with_kwargs=True
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles = []

    def forward(self, pixel_values):
        """Run CLIP and return the 4-layer cache + 2D coords.

        Returns
        -------
        cache : dict[name -> {"K": Tensor, "V": Tensor}]
            K, V have shape (B, vision_num_heads, T, vision_head_dim).
        coords : dict[name -> Tensor]
            Each entry has shape (B, T, 2).
        """
        self._captures = {}
        with self:
            _ = self.clip(pixel_values=pixel_values)
        B = pixel_values.shape[0]
        # Each capture has T = 1 + n_patches.  Drop the CLS row from
        # both K, V and the coords because the bridge operates on
        # patch tokens only.
        coords_full = self.patch_grid.unsqueeze(0).expand(B, -1, -1)
        cache = {}
        coords = {}
        for name, capture in self._captures.items():
            K = capture["K"][:, :, 1:, :]  # drop CLS
            V = capture["V"][:, :, 1:, :]
            cache[name] = {"K": K, "V": V}
            coords[name] = coords_full
        return cache, coords