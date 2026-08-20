"""Real-PyTorch tests for the cache_bridge_torch module.

These tests construct actual torch tensors with realistic shapes
(CLIP-ViT-L/14 + LLaMA-7B) and exercise the bridge's forward and
backward passes.  They need a working PyTorch install.

Run:
    python cache_bridge_torch/tests/test_bridge_real.py
"""

import sys
import os
import math
import math as _math

import torch
import torch.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARENT = os.path.dirname(ROOT)
sys.path.insert(0, PARENT)

from cache_bridge_torch.config import clip_llama_bridge_config
from cache_bridge_torch.bridge import (
    BridgeAdapter,
    DimensionProjection,
    StructurePreservingAttn,
    AdaptiveGating,
)
from cache_bridge_torch.inject import VisualInjector, simple_pos_tag


def test(name, fn):
    try:
        fn()
        print("  PASS:", name)
        return True
    except Exception as e:
        print("  FAIL:", name, "-", e)
        import traceback
        traceback.print_exc()
        return False


def fake_clip_cache(cfg, B=2, n_patches=None):
    """Build a fake 4-layer visual cache as if it came from CLIP."""
    cache = {}
    coords = {}
    n = n_patches if n_patches is not None else (cfg.image_size // cfg.patch_size) ** 2
    # Make a square grid that matches n
    side = int(math.sqrt(n))
    rows = torch.arange(side).repeat_interleave(side)
    cols = torch.arange(side).repeat(side)
    coords_full = torch.stack([rows, cols], dim=1).float()
    coords_full = coords_full.unsqueeze(0).expand(B, -1, -1)
    for name in ("basic", "semantic_object", "spatial_relation", "attribute"):
        K = torch.randn(B, cfg.vision_num_heads, n, cfg.vision_head_dim)
        V = torch.randn(B, cfg.vision_num_heads, n, cfg.vision_head_dim)
        cache[name] = {"K": K, "V": V}
        coords[name] = coords_full
    return cache, coords


def test_dimension_projection():
    cfg = clip_llama_bridge_config()
    proj = DimensionProjection(cfg)
    x = torch.randn(2, 16, cfg.vision_dim)
    y = proj(x)
    assert y.shape == (2, 16, cfg.language_dim), y.shape
    # Backward must work
    y.sum().backward()
    assert proj.down.weight.grad is not None
    assert proj.up.weight.grad is not None


def test_structure_preserving_attn():
    cfg = clip_llama_bridge_config()
    attn = StructurePreservingAttn(cfg)
    B, T = 2, 16
    x = torch.randn(B, T, cfg.language_dim)
    coords = torch.rand(B, T, 2) * (cfg.image_size // cfg.patch_size)
    y = attn(x, coords)
    assert y.shape == (B, T, cfg.language_dim), y.shape
    y.sum().backward()
    assert attn.bias.grad is not None


def test_adaptive_gating():
    cfg = clip_llama_bridge_config()
    gate = AdaptiveGating(cfg)
    B, T = 2, 16
    v = torch.randn(B, T, cfg.language_dim)
    t = torch.randn(B, T, cfg.language_dim)
    y = gate(v, t, layer_idx=0)
    assert y.shape == (B, T, cfg.language_dim)
    # When text_feat shape differs, broadcast
    t_short = torch.randn(B, 4, cfg.language_dim)
    y2 = gate(v, t_short)
    assert y2.shape == (B, T, cfg.language_dim)
    y.sum().backward()
    assert gate.gate_mlp.weight.grad is not None


def test_bridge_forward():
    cfg = clip_llama_bridge_config()
    bridge = BridgeAdapter(cfg)
    B = 2
    cache, coords = fake_clip_cache(cfg, B=B, n_patches=256)
    text_feats = torch.randn(B, 30, cfg.language_dim)
    layer_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])  # skip relation
    K, V = bridge(cache, text_feats, coords, layer_mask)
    # We activated basic + sem_obj + attribute = 3 layers, 256 patches each
    n_active = 3 * 256
    assert K.shape == (B, cfg.language_num_heads, n_active, cfg.language_head_dim), K.shape
    assert V.shape == (B, cfg.language_num_heads, n_active, cfg.language_head_dim), V.shape
    # Gradient must flow
    K.sum().backward()
    assert bridge.proj_k.down.weight.grad is not None


def test_bridge_mask_logic():
    """Different masks should produce different active token counts."""
    cfg = clip_llama_bridge_config()
    bridge = BridgeAdapter(cfg)
    B = 1
    cache, coords = fake_clip_cache(cfg, B=B, n_patches=64)
    text_feats = torch.randn(B, 16, cfg.language_dim)
    masks = [
        torch.tensor([1, 0, 0, 0], dtype=torch.float32),  # basic only
        torch.tensor([0, 1, 0, 0], dtype=torch.float32),  # object only
        torch.tensor([0, 0, 1, 0], dtype=torch.float32),  # relation only
        torch.tensor([0, 0, 0, 1], dtype=torch.float32),  # attribute only
        torch.tensor([1, 1, 1, 1], dtype=torch.float32),  # all
        torch.tensor([0, 0, 0, 0], dtype=torch.float32),  # none
    ]
    expected = [64, 64, 64, 64, 256, 0]
    for m, exp in zip(masks, expected):
        K, V = bridge(cache, text_feats, coords, m)
        assert K.shape[2] == exp, "mask %r gave K.shape=%s, expected %d" % (
            m.tolist(), K.shape, exp,
        )


def test_pos_tagger_real():
    """The simple_pos_tag should work on a small synthetic vocab."""
    # Build a tiny mock tokenizer that maps int -> str
    word_by_id = {
        100: "the", 101: "cat", 102: "sat", 103: "on",
        104: "red", 105: "mat", 106: "by", 107: "soft",
    }
    class MockTok:
        def decode(self, ids):
            return word_by_id.get(int(ids[0]), "")
    tok = MockTok()
    pos = simple_pos_tag([100, 101, 102, 103, 104, 105], tok)
    # the=0, cat=0, sat=0, on=1, red=2, mat=0
    assert pos == [0, 0, 0, 1, 2, 0], pos


def test_injector_prepend_splicing():
    """The injector's _prepend_basic should concatenate the basic
    layer before the text K, V at the right axis."""
    cfg = clip_llama_bridge_config()
    # Build a fake bridge output
    B, T_visual, T_text = 1, 64, 8
    K_bridge = torch.randn(B, cfg.language_num_heads, T_visual, cfg.language_head_dim)
    V_bridge = torch.randn(B, cfg.language_num_heads, T_visual, cfg.language_head_dim)
    K_text = torch.randn(B, cfg.language_num_heads, T_text, cfg.language_head_dim)
    V_text = torch.randn(B, cfg.language_num_heads, T_text, cfg.language_head_dim)
    # Build a dummy LLaMA model just for the injector
    class FakeLlama:
        class FakeModel:
            def __call__(self, *a, **kw):
                raise RuntimeError("not used in this test")
        model = FakeModel()
    injector = VisualInjector(cfg, FakeLlama(), bridge=None)
    new_K, new_V = injector._prepend_basic(K_bridge, V_bridge, T_visual, K_text, V_text)
    assert new_K.shape == (B, cfg.language_num_heads, T_text + T_visual, cfg.language_head_dim)
    assert new_V.shape == (B, cfg.language_num_heads, T_text + T_visual, cfg.language_head_dim)


def test_injector_interleave_splicing():
    """The injector's _interleave should insert one visual per text position."""
    cfg = clip_llama_bridge_config()
    B, T_text = 1, 4
    n_per_layer = 4
    T_visual = 4 * n_per_layer
    K_bridge = torch.randn(B, cfg.language_num_heads, T_visual, cfg.language_head_dim)
    V_bridge = torch.randn(B, cfg.language_num_heads, T_visual, cfg.language_head_dim)
    K_text = torch.randn(B, cfg.language_num_heads, T_text, cfg.language_head_dim)
    V_text = torch.randn(B, cfg.language_num_heads, T_text, cfg.language_head_dim)
    pos_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    class FakeLlama:
        class FakeModel:
            def __call__(self, *a, **kw):
                raise RuntimeError("not used in this test")
        model = FakeModel()
    injector = VisualInjector(cfg, FakeLlama(), bridge=None)
    new_K, new_V = injector._interleave(
        K_bridge, V_bridge,
        n_per_layer, n_per_layer, n_per_layer, n_per_layer,
        K_text, V_text, pos_ids,
    )
    # In interleave mode each text token is followed by a visual token
    assert new_K.shape == (B, cfg.language_num_heads, 2 * T_text, cfg.language_head_dim)
    assert new_V.shape == (B, cfg.language_num_heads, 2 * T_text, cfg.language_head_dim)


def test_full_forward_end_to_end():
    """End-to-end: build a fake LLaMA-like decoder, run the full forward
    with the bridge, check that the output shape is correct and that
    gradients flow through the bridge."""
    cfg = clip_llama_bridge_config()
    B, T_text = 1, 16
    bridge = BridgeAdapter(cfg)
    cache, coords = fake_clip_cache(cfg, B=B, n_patches=144)
    # Build a fake language model
    class FakeLMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(cfg.language_dim)
        def forward(self, x):
            return self.norm(x) + x
    class FakeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLMBlock() for _ in range(2)])
            self.token_emb = nn.Embedding(100, cfg.language_dim)
    fake_lm = FakeLM()
    # Use the bridge to produce K, V
    text_feats = fake_lm.token_emb(torch.randint(0, 100, (B, T_text)))
    layer_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    K_bridge, V_bridge = bridge(cache, text_feats, coords, layer_mask)
    # The output should have (B, 32, n_active, 128) shape
    n_active = 3 * 144
    assert K_bridge.shape == (B, cfg.language_num_heads, n_active, cfg.language_head_dim)
    # A simple "attention-like" op: softmax(QK^T) V
    Q = text_feats.view(B, T_text, cfg.language_num_heads, cfg.language_head_dim).transpose(1, 2)
    attn = torch.matmul(Q, K_bridge.transpose(-2, -1)) / math.sqrt(cfg.language_head_dim)
    p = torch.softmax(attn, dim=-1)
    out = torch.matmul(p, V_bridge).transpose(1, 2).reshape(B, T_text, cfg.language_dim)
    assert out.shape == (B, T_text, cfg.language_dim)
    out.sum().backward()
    # Gradients should reach the bridge params
    assert bridge.proj_k.down.weight.grad is not None
    assert bridge.attn.bias.grad is not None
    assert bridge.gate.gate_mlp.weight.grad is not None
    print("    bridge trainable params with grad:")
    n = 0
    for p in bridge.parameters():
        if p.grad is not None:
            n += p.numel()
    print("    total grad params:", n)


def test_param_count():
    """The LoRA bridge should be much smaller than a full projection."""
    cfg = clip_llama_bridge_config()
    bridge = BridgeAdapter(cfg)
    n = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    full = cfg.vision_dim * cfg.language_dim
    print("    bridge trainable params: %d (%.2f%% of full projection %d)" %
          (n, 100.0 * n / full, full))
    assert n < 0.50 * full  # small enough to train


def main():
    tests = [
        ("dimension_projection", test_dimension_projection),
        ("structure_preserving_attn", test_structure_preserving_attn),
        ("adaptive_gating", test_adaptive_gating),
        ("bridge_forward", test_bridge_forward),
        ("bridge_mask_logic", test_bridge_mask_logic),
        ("pos_tagger_real", test_pos_tagger_real),
        ("injector_prepend_splicing", test_injector_prepend_splicing),
        ("injector_interleave_splicing", test_injector_interleave_splicing),
        ("full_forward_end_to_end", test_full_forward_end_to_end),
        ("param_count", test_param_count),
    ]
    n_pass = 0
    n_total = len(tests)
    print("Running %d real-PyTorch tests" % n_total)
    for name, fn in tests:
        if test(name, fn):
            n_pass += 1
    print("%d / %d tests passed" % (n_pass, n_total))
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()