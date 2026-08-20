# -*- coding: utf-8 -*-
"""Configuration for the cache-bridging system."""

from collections import namedtuple

ModelConfig = namedtuple("ModelConfig", [
    "image_size",
    "patch_size",
    "vision_dim",
    "language_dim",
    "num_basic_tokens",
    "num_object_tokens",
    "num_relation_tokens",
    "num_attribute_tokens",
    "num_decoder_layers",
    "num_heads",
    "ffn_hidden",
    "max_seq_len",
    "vocab_size",
    "dropout",
    "compression_keep_ratio",
    "compression_sim_threshold",
    "num_query_types",
])


def default_config():
    """Return a small configuration suitable for a CPU smoke test."""
    return ModelConfig(
        image_size=32,
        patch_size=8,
        vision_dim=64,
        language_dim=64,
        num_basic_tokens=8,
        num_object_tokens=6,
        num_relation_tokens=4,
        num_attribute_tokens=6,
        num_decoder_layers=2,
        num_heads=4,
        ffn_hidden=128,
        max_seq_len=64,
        vocab_size=256,
        dropout=0.0,
        compression_keep_ratio=0.5,
        compression_sim_threshold=0.9999,
        num_query_types=3,
    )


def small_demo_config():
    """A larger configuration more representative of a real model."""
    return ModelConfig(
        image_size=64,
        patch_size=8,
        vision_dim=128,
        language_dim=128,
        num_basic_tokens=16,
        num_object_tokens=12,
        num_relation_tokens=8,
        num_attribute_tokens=12,
        num_decoder_layers=3,
        num_heads=4,
        ffn_hidden=256,
        max_seq_len=128,
        vocab_size=512,
        dropout=0.0,
        compression_keep_ratio=0.5,
        compression_sim_threshold=0.9999,
        num_query_types=3,
    )

def tiny_smoke_config():
    """A minimal config for the smoke test (CPU friendly)."""
    return ModelConfig(
        image_size=16, patch_size=8,
        vision_dim=8, language_dim=8,
        num_basic_tokens=2, num_object_tokens=2, num_relation_tokens=2, num_attribute_tokens=2,
        num_decoder_layers=1, num_heads=2, ffn_hidden=16,
        max_seq_len=32, vocab_size=128, dropout=0.0,
        compression_keep_ratio=0.5, compression_sim_threshold=0.9999,
        num_query_types=3,
    )