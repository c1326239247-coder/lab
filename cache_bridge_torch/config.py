# -*- coding: utf-8 -*-
"""Configuration for the CLIP + LLaMA bridge adapter.

The shapes are aligned to public Hugging Face model checkpoints:

  openai/clip-vit-large-patch14
      vision encoder  : 24 layers
      vision_dim      : 1024
      num_heads       : 16
      head_dim        : 64
      image_size      : 224
      patch_size      : 14
      n_patches       : 16 * 16 = 256  (+ CLS = 257)

  meta-llama/Llama-2-7b-hf
      text decoder   : 32 layers
      language_dim   : 4096
      num_heads      : 32
      head_dim       : 128
      vocab_size     : 32000
      max_seq_len    : 4096

For other checkpoints just override the corresponding fields.  The
bridge itself is checkpoint-agnostic.
"""

from collections import namedtuple


BridgeConfig = namedtuple("BridgeConfig", [
    # CLIP side
    "vision_dim",
    "vision_num_heads",
    "vision_head_dim",
    "vision_num_layers",
    "image_size",
    "patch_size",
    # Which CLIP layers to tap for the 4 cache categories.
    # Indices into the CLIP encoder layers.
    "basic_layer_idx",          # early
    "object_layer_idx",         # mid
    "relation_layer_idx",       # late
    "attribute_layer_idx",      # late
    # LLaMA side
    "language_dim",
    "language_num_heads",
    "language_head_dim",
    "language_num_layers",
    "max_seq_len",
    # Bridge hyperparameters
    "bridge_rank",              # LoRA rank for the dimension projection
    "bridge_dropout",
    "gate_hidden",
])


def clip_llama_bridge_config():
    """Default configuration for ViT-L/14 + LLaMA-7B."""
    return BridgeConfig(
        vision_dim=1024,
        vision_num_heads=16,
        vision_head_dim=64,
        vision_num_layers=24,
        image_size=224,
        patch_size=14,
        basic_layer_idx=4,
        object_layer_idx=12,
        relation_layer_idx=18,
        attribute_layer_idx=22,
        language_dim=4096,
        language_num_heads=32,
        language_head_dim=128,
        language_num_layers=32,
        max_seq_len=4096,
        bridge_rank=64,
        bridge_dropout=0.0,
        gate_hidden=32,
)