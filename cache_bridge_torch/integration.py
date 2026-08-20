# -*- coding: utf-8 -*-
"""High-level integration with Hugging Face transformers.

This is the file you would import in a real research script.  It ties
the extractor, the bridge, and the injector together with a single
function call.

Example
-------
    from cache_bridge_torch.integration import build_cache_bridge_llm
    from transformers import (
        CLIPVisionModel, CLIPProcessor,
        LlamaForCausalLM, LlamaTokenizer,
    )

    clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    llama = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tok = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    model = build_cache_bridge_llm(cfg, clip, llama)
    model.set_image(proc(images=image, return_tensors="pt")["pixel_values"])
    model.set_prompt(tok(prompt, return_tensors="pt")["input_ids"][0])
    out = model.generate(max_new_tokens=64)
"""

import torch
import torch.nn as nn

from .bridge import BridgeAdapter
from .extract import CLIPHierarchicalCacheExtractor
from .inject import VisualInjector, simple_pos_tag


class CacheBridgeLLM(nn.Module):
    """The full CLIP + Bridge + LLaMA stack, wrapped in a single object."""

    def __init__(self, cfg, clip_vision_model, llama_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.clip_extractor = CLIPHierarchicalCacheExtractor(cfg, clip_vision_model)
        self.bridge = BridgeAdapter(cfg)
        self.injector = VisualInjector(cfg, llama_model, self.bridge)
        self.tokenizer = tokenizer
        # Defaults: the full layer mask (all four layers active)
        self.register_buffer(
            "default_layer_mask",
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
        # Track how many tokens each layer contributes, set after
        # set_image is called.
        self._layer_token_counts = None
        # Stash the layer order so the injector can split the bridged
        # cache into the four sub-caches.
        self.cfg._bridge_layer_order = list(self.bridge.LAYER_NAMES)

    def set_image(self, pixel_values):
        """Run CLIP and cache the hierarchical visual KV."""
        cache, coords = self.clip_extractor(pixel_values)
        # Record how many tokens each layer actually has.
        self._layer_token_counts = [
            cache[name]["K"].shape[2] for name in self.bridge.LAYER_NAMES
        ]
        self._last_cache = cache
        self._last_coords = coords
        return cache

    def set_prompt(self, input_ids):
        """Tokenize a prompt and figure out the per-position slot types."""
        pos = simple_pos_tag(input_ids.tolist(), self.tokenizer)
        self._last_pos = torch.tensor(pos, dtype=torch.long)
        return self._last_pos

    def set_layer_mask(self, mask):
        """Override the layer mask (e.g. for spatial_reasoning queries)."""
        self._last_mask = torch.tensor(mask, dtype=torch.float32)

    def forward(self, input_ids, **kwargs):
        layer_mask = getattr(self, "_last_mask", self.default_layer_mask)
        self.injector.set_visual_cache(
            self._last_cache, self._last_coords, layer_mask, self._last_pos,
        )
        return self.injector(input_ids, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, **gen_kwargs):
        layer_mask = getattr(self, "_last_mask", self.default_layer_mask)
        self.injector.set_visual_cache(
            self._last_cache, self._last_coords, layer_mask, self._last_pos,
        )
        return self.injector.llama.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )


def build_cache_bridge_llm(cfg, clip_vision_model, llama_model, tokenizer):
    """Convenience factory.  See the module docstring for an example."""
    return CacheBridgeLLM(cfg, clip_vision_model, llama_model, tokenizer)