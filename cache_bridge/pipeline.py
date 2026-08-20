# -*- coding: utf-8 -*-
"""End-to-end pipeline that wires all the modules together.

  Image ---> VisualEncoder ---> HierarchicalKVCache
                                       |
                                       v
                              CompressionModule (optional)
                                       |
                                       v
                          +-------- BridgeAdapter ---------+
                          |                                |
                          v                                v
                DynamicRouter -> layer_mask     (K, V) -> LanguageModel
                          |                                |
                          v                                v
                position_id (per text pos)        next-token logits
"""

import numpy as np

from .visual_encoder import VisualEncoder
from .bridge_adapter import BridgeAdapter
from .language_model import LanguageModel
from .dynamic_router import DynamicRouter
from .compression import CompressionModule
from .hierarchical_kv import HierarchicalKVCache


class CacheBridgePipeline(object):
    """The full system in a single object."""

    def __init__(self, config, seed=0, use_compression=True):
        self.config = config
        self.encoder = VisualEncoder(config, seed=seed)
        self.adapter = BridgeAdapter(config, seed=seed + 1)
        self.lm = LanguageModel(config, seed=seed + 2)
        self.router = DynamicRouter(config.language_dim, seed=seed + 3)
        self.compression = CompressionModule(
            keep_ratio=config.compression_keep_ratio,
            sim_threshold=config.compression_sim_threshold,
        ) if use_compression else None

    def encode_image(self, image):
        cache = self.encoder.encode(image)
        if self.compression is not None:
            cache = self.compression.compress(cache)
        return cache

    def forward(self, image, query_token_ids, position_id):
        cache = self.encode_image(image)
        # Build a query embedding by averaging the token embeddings.
        if len(query_token_ids) == 0:
            q_emb = self.encoder.patch_encoder.cls[0]
        else:
            q_emb = self.lm.token_emb[query_token_ids].mean(axis=0)
        layer_mask, pos_weights, gate_bias = self.router(q_emb)
        # Get the text features the bridge adapter will mix with.
        text_feats = self.lm.embed(query_token_ids[None])[0]
        K, V = self.adapter(cache, text_feats, layer_mask)
        # Per-position type from the position_id argument.
        pos = position_id[None]
        logits = self.lm.forward(query_token_ids[None], K, V, pos)
        return logits, layer_mask, pos_weights, gate_bias

    def generate(self, image, prompt_ids, position_id, max_new_tokens=20):
        cache = self.encode_image(image)
        if len(prompt_ids) == 0:
            q_emb = self.encoder.patch_encoder.cls[0]
        else:
            q_emb = self.lm.token_emb[prompt_ids].mean(axis=0)
        layer_mask, pos_weights, gate_bias = self.router(q_emb)
        text_feats = self.lm.embed(prompt_ids[None])[0]
        K, V = self.adapter(cache, text_feats, layer_mask)
        pos = position_id[None]
        return self.lm.greedy_generate(prompt_ids, K, V, pos, max_new_tokens=max_new_tokens)

    def summary(self):
        layers = (
            "basic=%d sem_obj=%d spatial=%d attr=%d"
            % (
                self.config.num_basic_tokens,
                self.config.num_object_tokens,
                self.config.num_relation_tokens,
                self.config.num_attribute_tokens,
            )
        )
        return (
            "CacheBridgePipeline(vision_dim=%d, language_dim=%d, %s, "
            "decoder_layers=%d, vocab=%d, max_seq=%d)"
            % (
                self.config.vision_dim,
                self.config.language_dim,
                layers,
                self.config.num_decoder_layers,
                self.config.vocab_size,
                self.config.max_seq_len,
            )
        )