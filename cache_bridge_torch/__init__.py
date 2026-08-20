# -*- coding: utf-8 -*-
"""Public API for the cache-bridging CLIP + LLaMA integration."""

from .config import BridgeConfig, clip_llama_bridge_config
from .bridge import (
    BridgeAdapter,
    DimensionProjection,
    StructurePreservingAttn,
    AdaptiveGating,
)
from .extract import CLIPHierarchicalCacheExtractor
from .inject import VisualInjector, simple_pos_tag
from .integration import CacheBridgeLLM, build_cache_bridge_llm


__all__ = [
    "BridgeConfig",
    "clip_llama_bridge_config",
    "DimensionProjection",
    "StructurePreservingAttn",
    "AdaptiveGating",
    "BridgeAdapter",
    "CLIPHierarchicalCacheExtractor",
    "VisualInjector",
    "simple_pos_tag",
    "CacheBridgeLLM",
    "build_cache_bridge_llm",
]