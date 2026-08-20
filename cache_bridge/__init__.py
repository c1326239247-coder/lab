# -*- coding: utf-8 -*-
"""Cache-Bridging Cross-Modal Understanding System.

NumPy-only reproduction of the architecture described in the patent
"Cache-bridging cross-modal understanding system".  The system routes
a hierarchical visual key-value cache through a learned bridge adapter
into a language model, replacing the lossy "image -> text -> LLM"
pipeline with a lossless "image -> KV -> LLM" pipeline that preserves
spatial, semantic, and attribute information.
"""

from .config import (
    ModelConfig,
    default_config,
    small_demo_config,
    tiny_smoke_config,
)
from .hierarchical_kv import HierarchicalKVCache, LayerKV
from .visual_encoder import VisualEncoder
from .bridge_adapter import BridgeAdapter
from .language_model import LanguageModel
from .dynamic_router import DynamicRouter
from .compression import CompressionModule
from .data import SyntheticMultimodalDataset
from .pipeline import CacheBridgePipeline
from .training import Trainer


__all__ = [
    "ModelConfig",
    "default_config",
    "small_demo_config",
    "tiny_smoke_config",
    "HierarchicalKVCache",
    "LayerKV",
    "VisualEncoder",
    "BridgeAdapter",
    "LanguageModel",
    "DynamicRouter",
    "CompressionModule",
    "SyntheticMultimodalDataset",
    "CacheBridgePipeline",
    "Trainer",
]