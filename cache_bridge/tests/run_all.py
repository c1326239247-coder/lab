# -*- coding: utf-8 -*-
"""Smoke tests for the cache-bridging system.

Run with:
    cd <repo>
    py cache_bridge/tests/run_all.py
"""

import os
import sys
import traceback
import numpy as np

# Make the parent cache_bridge package importable.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(ROOT))

from cache_bridge.config import default_config, tiny_smoke_config
from cache_bridge.hierarchical_kv import HierarchicalKVCache, LayerKV
from cache_bridge.visual_encoder import VisualEncoder
from cache_bridge.bridge_adapter import BridgeAdapter
from cache_bridge.language_model import LanguageModel
from cache_bridge.dynamic_router import DynamicRouter
from cache_bridge.compression import CompressionModule, topk_select, cosine_merge
from cache_bridge.data import SyntheticMultimodalDataset
from cache_bridge.pipeline import CacheBridgePipeline
from cache_bridge.training import Trainer


def test(name, fn):
    try:
        fn()
        print("  PASS:", name)
        return True
    except Exception:
        print("  FAIL:", name)
        traceback.print_exc()
        return False


def test_hierarchical_kv():
    cfg = default_config()
    rng = np.random.RandomState(0)
    layers = {}
    for name in HierarchicalKVCache.LAYER_NAMES:
        n = cfg.num_basic_tokens
        layers[name] = LayerKV(
            keys=rng.randn(n, cfg.vision_dim).astype(np.float32),
            values=rng.randn(n, cfg.vision_dim).astype(np.float32),
            coords=rng.uniform(0, 1, (n, 2)).astype(np.float32),
            meta=["t_%d" % i for i in range(n)],
            layer_name=name,
        )
    cache = HierarchicalKVCache(
        layers["basic"], layers["semantic_object"],
        layers["spatial_relation"], layers["attribute"],
    )
    assert cache.total_tokens() == sum(len(l) for l in layers.values())
    K, V, C, L, M = cache.concat_tokens()
    assert K.shape[0] == cache.total_tokens()
    assert len(M) == K.shape[0]
    mask = [1, 0, 1, 0]
    sub = cache.select_layers(mask)
    assert len(sub.get("basic")) == cfg.num_basic_tokens
    assert len(sub.get("semantic_object")) == 0


def test_visual_encoder():
    cfg = default_config()
    enc = VisualEncoder(cfg, seed=0)
    img = (np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8)
    cache = enc.encode(img)
    assert len(cache.get("basic")) == cfg.num_basic_tokens
    assert len(cache.get("semantic_object")) == cfg.num_object_tokens
    assert len(cache.get("spatial_relation")) == cfg.num_relation_tokens
    assert len(cache.get("attribute")) == cfg.num_attribute_tokens
    for name in HierarchicalKVCache.LAYER_NAMES:
        l = cache.get(name)
        assert l.keys.shape[1] == cfg.vision_dim
        assert l.coords.shape == (l.keys.shape[0], 2)


def test_bridge_adapter():
    cfg = default_config()
    enc = VisualEncoder(cfg, seed=0)
    bridge = BridgeAdapter(cfg, seed=1)
    img = (np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8)
    cache = enc.encode(img)
    text_feats = np.random.randn(30, cfg.language_dim).astype(np.float32)
    layer_mask = np.array([1, 1, 0, 1], dtype=np.float32)
    K, V = bridge(cache, text_feats, layer_mask)
    n = sum(len(cache.get(n)) for i, n in enumerate(HierarchicalKVCache.LAYER_NAMES) if layer_mask[i])
    assert K.shape == (n, cfg.language_dim)
    assert V.shape == (n, cfg.language_dim)


def test_language_model():
    cfg = default_config()
    lm = LanguageModel(cfg, seed=0)
    ids = np.array([[10, 20, 30, 40]], dtype=np.int64)
    logits = lm.forward(ids)
    assert logits.shape == (1, 4, cfg.vocab_size)
    # With visual cache
    K = np.random.randn(5, cfg.language_dim).astype(np.float32)
    V = np.random.randn(5, cfg.language_dim).astype(np.float32)
    pos = np.array([[0, 0, 1, 0]], dtype=np.int64)
    logits2 = lm.forward(ids, K, V, pos)
    assert logits2.shape == (1, 4, cfg.vocab_size)
    out = lm.greedy_generate(np.array([10, 20, 30], dtype=np.int64), K, V, pos, max_new_tokens=3)
    assert out.shape[0] == 6


def test_dynamic_router():
    r = DynamicRouter(query_dim=16, seed=0)
    emb = np.random.randn(16).astype(np.float32)
    mask, pos, gate = r(emb)
    assert mask.shape == (4,)
    assert pos.shape == (3,)
    assert 0.0 <= gate <= 1.0


def test_compression():
    keys = np.random.randn(8, 4).astype(np.float32)
    values = np.random.randn(8, 4).astype(np.float32)
    coords = np.random.uniform(0, 1, (8, 2)).astype(np.float32)
    k2, v2, c2 = topk_select(keys, values, coords, 0.5)
    assert k2.shape[0] == 4
    # Make values highly similar so the merge kicks in
    values2 = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (8, 1)).astype(np.float32)
    v3, c3 = cosine_merge(values2, coords, 0.5)
    assert v3.shape[0] < values2.shape[0]
    # End-to-end on a full cache
    cfg = default_config()
    enc = VisualEncoder(cfg, seed=0)
    cache = enc.encode((np.random.rand(cfg.image_size, cfg.image_size, 3) * 255).astype(np.uint8))
    cm = CompressionModule(keep_ratio=0.5, sim_threshold=0.95)
    new = cm.compress(cache)
    for name in HierarchicalKVCache.LAYER_NAMES:
        old = cache.get(name)
        newl = new.get(name)
        assert newl.keys.shape[1] == old.keys.shape[1]
        assert newl.coords.shape[1] == old.coords.shape[1]


def test_dataset():
    cfg = default_config()
    ds = SyntheticMultimodalDataset(3, cfg, seed=0)
    assert len(ds) == 3
    ex = ds[0]
    assert ex["image"].shape == (cfg.image_size, cfg.image_size, 3)
    assert len(ex["query_token_ids"]) > 0
    assert len(ex["response_token_ids"]) > 0
    assert len(ex["position_id"]) == len(ex["query_token_ids"])


def test_pipeline():
    cfg = default_config()
    p = CacheBridgePipeline(cfg, seed=0)
    ds = SyntheticMultimodalDataset(2, cfg, seed=0)
    ex = ds[0]
    logits, mask, pos, gate = p.forward(ex["image"], ex["query_token_ids"], ex["position_id"])
    assert logits.shape[0] == 1
    assert logits.shape[1] == len(ex["query_token_ids"])
    assert logits.shape[2] == cfg.vocab_size
    out = p.generate(ex["image"], ex["query_token_ids"], ex["position_id"], max_new_tokens=5)
    assert out.shape[0] == len(ex["query_token_ids"]) + 5


def test_trainer_tiny():
    """A 2-iteration training run with the tiny smoke config to make sure
    the 3-stage loop does not crash on the smallest possible model."""
    cfg = tiny_smoke_config()
    pipeline = CacheBridgePipeline(cfg, seed=0)
    dataset = SyntheticMultimodalDataset(2, cfg, seed=0)
    trainer = Trainer(pipeline, dataset, cfg)
    trainer.run(stage1_iters=1, stage2_iters=1, stage3_iters=1)


def main():
    tests = [
        ("hierarchical_kv", test_hierarchical_kv),
        ("visual_encoder", test_visual_encoder),
        ("bridge_adapter", test_bridge_adapter),
        ("language_model", test_language_model),
        ("dynamic_router", test_dynamic_router),
        ("compression", test_compression),
        ("dataset", test_dataset),
        ("pipeline", test_pipeline),
        ("trainer_tiny", test_trainer_tiny),
    ]
    n_pass = 0
    n_total = len(tests)
    print("Running %d tests" % n_total)
    for name, fn in tests:
        if test(name, fn):
            n_pass += 1
    print("%d / %d tests passed" % (n_pass, n_total))
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()