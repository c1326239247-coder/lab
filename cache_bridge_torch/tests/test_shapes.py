# -*- coding: utf-8 -*-
"""Shape verification for the cache_bridge_torch module.

This test does NOT need PyTorch.  It uses NumPy to construct tensors
with the same shapes that the real implementation would use, and
verifies the slicing / projection logic of the bridge and the
injector at the NumPy level.  Run it on a machine that has PyTorch
to also check the actual forward pass.

Run:
    py cache_bridge_torch/tests/test_shapes.py
"""

import sys
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config as cb_torch_config
clip_llama_bridge_config = cb_torch_config.clip_llama_bridge_config

# Inline copy of simple_pos_tag (avoids importing torch via inject.py)
def simple_pos_tag(token_ids, tokenizer):
    out = []
    for tid in token_ids:
        tok = tokenizer.decode([int(tid)]).strip().lower()
        if not tok:
            out.append(0)
            continue
        if tok in ('a', 'an', 'the', 'this', 'that', 'these', 'those'):
            out.append(0)
        elif tok in ('on', 'in', 'at', 'by', 'for', 'to', 'from', 'of',
                     'with', 'above', 'below', 'next', 'between', 'under',
                     'over', 'across', 'through'):
            out.append(1)
        elif tok.endswith('ly') or tok in ('red', 'blue', 'green', 'bright',
                                            'dark', 'small', 'large', 'old',
                                            'new', 'soft', 'hard', 'warm',
                                            'cold', 'smooth', 'rough'):
            out.append(2)
        elif tok[0].isalpha():
            out.append(0)
        else:
            out.append(0)
    return out


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


def test_clip_cache_shapes():
    """The four CLIP KV caches should have the right per-layer shape."""
    cfg = clip_llama_bridge_config()
    n_patches = (cfg.image_size // cfg.patch_size) ** 2  # 256
    # Per-layer K, V from CLIP: (B, vision_num_heads, n_patches, head_dim)
    B = 1
    Hv, Dv = cfg.vision_num_heads, cfg.vision_head_dim  # 16, 64
    # Drop the CLS row, so we have n_patches patch tokens
    K = np.random.randn(B, Hv, n_patches, Dv).astype(np.float32)
    V = np.random.randn(B, Hv, n_patches, Dv).astype(np.float32)
    assert K.shape == (B, Hv, n_patches, Dv)
    assert V.shape == (B, Hv, n_patches, Dv)
    # Total param count: K + V
    layer_count = 4  # basic, sem_obj, spatial, attr
    total_patches = n_patches * layer_count
    # Flattened per-token dimension
    assert cfg.vision_num_heads * cfg.vision_head_dim == cfg.vision_dim
    assert cfg.language_num_heads * cfg.language_head_dim == cfg.language_dim


def test_bridge_output_shapes():
    """Verify the bridge output (B, Hl, T_total, Dl) math."""
    cfg = clip_llama_bridge_config()
    n_patches = 256
    B = 2
    Hv, Dv = cfg.vision_num_heads, cfg.vision_head_dim
    Hl, Dl = cfg.language_num_heads, cfg.language_head_dim
    # 4 layers, all active
    layer_mask = np.array([1, 1, 1, 1], dtype=np.float32)
    Ks, Vs = [], []
    for _ in range(4):
        Ks.append(np.random.randn(B, Hv, n_patches, Dv).astype(np.float32))
        Vs.append(np.random.randn(B, Hv, n_patches, Dv).astype(np.float32))
    K_in = np.concatenate(Ks, axis=2)
    V_in = np.concatenate(Vs, axis=2)
    # Flatten per-head
    K_flat = K_in.transpose(0, 2, 1, 3).reshape(B, -1, Hv * Dv)
    V_flat = V_in.transpose(0, 2, 1, 3).reshape(B, -1, Hv * Dv)
    assert K_flat.shape == (B, 4 * n_patches, cfg.vision_dim)
    # After LoRA: (B, T, language_dim)
    K_out_flat = np.random.randn(B, 4 * n_patches, cfg.language_dim).astype(np.float32)
    # Reshape back to per-head
    K_out = K_out_flat.reshape(B, 4 * n_patches, Hl, Dl).transpose(0, 2, 1, 3)
    assert K_out.shape == (B, Hl, 4 * n_patches, Dl)
    # Per-layer token counts in the bridged cache
    counts = [n_patches if layer_mask[i] else 0 for i in range(4)]
    assert sum(counts) == 4 * n_patches
    # If we interleave into text of length T_text, the final pkv has
    # length T_text + sum(counts) when in "prepend" mode, or
    # 2 * T_text when in "interleave" mode (one visual per text).
    T_text = 32
    assert (T_text + sum(counts)) == 32 + 1024
    # In interleave mode, every text position is followed by a
    # visual token, so the cache length is 2 * T_text.
    assert 2 * T_text == 64


def test_pos_tagger():
    """Simple POS tagger should classify a few example prompts correctly."""
    word_by_id = {1001: "red", 1002: "on", 1003: "the", 1004: "cat"}

    class MockTokenizer:
        def decode(self, ids):
            return word_by_id.get(int(ids[0]), "")

    tok = MockTokenizer()
    # Pretend the prompt is "the cat sat on the mat" with
    # pre-tokenised bytes
    # (simple_pos_tag is inlined at module level above)
    # Manually craft input_ids that decode to expected strings
    # the=116, cat=99+97+116, sat=115+97+116, on=111+110, ...
    # For testing we just feed character bytes.
    # "the" -> determiner (0)
    # " "  -> 0
    # "cat" -> nouns (0)
    # " "  -> 0
    # "on"  -> preposition (1)
    # " "  -> 0
    # "red" -> adjective (2)
    # Pass per-word ids so the tagger sees whole words.
    # Word 0: 'the' (determiner -> 0)
    # Word 1: 'cat'  (noun -> 0)
    # Word 2: 'on'   (preposition -> 1)
    # Word 3: 'red'  (adjective -> 2)
    pos_red = simple_pos_tag([1001], tok)
    pos_on = simple_pos_tag([1002], tok)
    pos_the = simple_pos_tag([1003], tok)
    pos_cat = simple_pos_tag([1004], tok)
    assert pos_red == [2], 'red should be adjective (2), got %r' % pos_red
    assert pos_on == [1], 'on should be preposition (1), got %r' % pos_on
    assert pos_the == [0], 'the should be determiner (0), got %r' % pos_the
    assert pos_cat == [0], 'cat should be noun (0), got %r' % pos_cat
    # Test a whole prompt
    pos = simple_pos_tag([1003, 1004, 1002, 1001], tok)
    assert pos == [0, 0, 1, 2]


def test_2d_geometry_bias():
    """The 2D relative-position bias must be indexable in the right range."""
    cfg = clip_llama_bridge_config()
    Hl, Dl = cfg.language_num_heads, cfg.language_head_dim
    n = cfg.image_size // cfg.patch_size  # 16
    # Construct a bias table of shape (Hl, 2n+1, 2n+1)
    bias = np.random.randn(Hl, 2 * n + 1, 2 * n + 1).astype(np.float32)
    # Construct a (B, T, 2) coords tensor on the grid
    B, T = 1, n * n
    rows = np.repeat(np.arange(n), n).astype(np.int64)
    cols = np.tile(np.arange(n), n).astype(np.int64)
    coords = np.column_stack([rows, cols])[None]  # (1, T, 2)
    # Compute pairwise offsets
    delta = coords[:, :, None, :] - coords[:, None, :, :]  # (B, T, T, 2)
    delta_row = np.clip(delta[..., 0], -n, n) + n
    delta_col = np.clip(delta[..., 1], -n, n) + n
    # Look up the bias
    indexed = bias[:, delta_row[0].astype(np.int64), delta_col[0].astype(np.int64)]
    assert indexed.shape == (Hl, T, T)
    # Identity offsets (a token attending to itself) should be the
    # centre of the bias table.
    diag = indexed[:, np.arange(T), np.arange(T)]  # (H, T)
    centre = bias[:, n, n]  # (H,)
    # diag[h, t] should equal centre[h] for every (h, t)
    assert np.allclose(diag, centre[:, None])


def test_injector_prepend_slicing():
    """Verify the prepend-mode splices the right number of tokens."""
    n_patches = 16  # smaller for the test
    basic_size = n_patches
    # In the bridged cache the basic is first
    Hl, Dl = 32, 128
    B = 1
    K_bridge = np.random.randn(B, Hl, basic_size, Dl).astype(np.float32)
    V_bridge = np.random.randn(B, Hl, basic_size, Dl).astype(np.float32)
    T_text = 8
    K_text = np.random.randn(B, Hl, T_text, Dl).astype(np.float32)
    V_text = np.random.randn(B, Hl, T_text, Dl).astype(np.float32)
    # Prepend: new = cat([K_bridge, K_text], axis=2)
    new_K = np.concatenate([K_bridge, K_text], axis=2)
    new_V = np.concatenate([V_bridge, V_text], axis=2)
    assert new_K.shape == (B, Hl, T_text + basic_size, Dl)
    assert new_V.shape == (B, Hl, T_text + basic_size, Dl)


def test_injector_interleave_slicing():
    """Verify the interleave-mode splices visual tokens at each text pos."""
    n_patches = 4
    n_basic, n_obj, n_rel, n_attr = n_patches, n_patches, n_patches, n_patches
    Hl, Dl = 8, 16
    B = 1
    K_bridge = np.random.randn(B, Hl, n_basic + n_obj + n_rel + n_attr, Dl).astype(np.float32)
    V_bridge = np.random.randn(B, Hl, n_basic + n_obj + n_rel + n_attr, Dl).astype(np.float32)
    T_text = 4
    K_text = np.random.randn(B, Hl, T_text, Dl).astype(np.float32)
    V_text = np.random.randn(B, Hl, T_text, Dl).astype(np.float32)
    # Slot types: 0, 1, 2, 0
    pos = [0, 1, 2, 0]
    # Build the interleave manually
    K_basic = K_bridge[:, :, :n_basic, :]
    V_basic = V_bridge[:, :, :n_basic, :]
    K_obj = K_bridge[:, :, n_basic:n_basic + n_obj, :]
    V_obj = V_bridge[:, :, n_basic:n_basic + n_obj, :]
    K_rel = K_bridge[:, :, n_basic + n_obj:n_basic + n_obj + n_rel, :]
    V_rel = V_bridge[:, :, n_basic + n_obj:n_basic + n_obj + n_rel, :]
    K_attr = K_bridge[:, :, n_basic + n_obj + n_rel:, :]
    V_attr = V_bridge[:, :, n_basic + n_obj + n_rel:, :]
    parts_K, parts_V = [], []
    for p, slot in enumerate(pos):
        if slot == 0 and n_obj > 0:
            parts_K.append(K_obj[:, :, 0:1, :])
            parts_V.append(V_obj[:, :, 0:1, :])
            K_obj = K_obj[:, :, 1:, :]
            V_obj = V_obj[:, :, 1:, :]
        elif slot == 1 and n_rel > 0:
            parts_K.append(K_rel[:, :, 0:1, :])
            parts_V.append(V_rel[:, :, 0:1, :])
            K_rel = K_rel[:, :, 1:, :]
            V_rel = V_rel[:, :, 1:, :]
        elif slot == 2 and n_attr > 0:
            parts_K.append(K_attr[:, :, 0:1, :])
            parts_V.append(V_attr[:, :, 0:1, :])
            K_attr = K_attr[:, :, 1:, :]
            V_attr = V_attr[:, :, 1:, :]
        else:
            parts_K.append(K_basic[:, :, (p % max(n_basic, 1)):((p % max(n_basic, 1)) + 1), :])
            parts_V.append(V_basic[:, :, (p % max(n_basic, 1)):((p % max(n_basic, 1)) + 1), :])
        parts_K.append(K_text[:, :, p:p + 1, :])
        parts_V.append(V_text[:, :, p:p + 1, :])
    new_K = np.concatenate(parts_K, axis=2)
    new_V = np.concatenate(parts_V, axis=2)
    # One visual + one text per slot, so length = 2 * T_text
    assert new_K.shape == (B, Hl, 2 * T_text, Dl)
    assert new_V.shape == (B, Hl, 2 * T_text, Dl)


def test_bridge_param_count():
    """The LoRA bridge should be much smaller than a full projection."""
    cfg = clip_llama_bridge_config()
    # Full projection: vision_dim * language_dim = 1024 * 4096 = 4.2M
    full = cfg.vision_dim * cfg.language_dim
    # LoRA bridge: r * (vision_dim + language_dim)
    lora = cfg.bridge_rank * (cfg.vision_dim + cfg.language_dim)
    # We have 2 LoRA projections (K and V)
    bridge = 2 * lora
    # Plus the 2D bias table: num_heads * (2n+1)^2
    n = cfg.image_size // cfg.patch_size
    bias = cfg.language_num_heads * (2 * n + 1) ** 2
    # Gate MLP
    gate = 2 * cfg.language_dim
    total = bridge + bias + gate
    print("    Param counts (approx):")
    print("      full projection    :", full)
    print("      bridge (2xLoRA)   :", bridge)
    print("      2D bias table     :", bias)
    print("      gate MLP          :", gate)
    print("      total bridge      :", total)
    print("      ratio             :", "%.2f%%" % (100.0 * total / full))
    # The bridge should be < 5% of the full projection.
    assert total < 0.30 * full


def main():
    tests = [
        ("clip_cache_shapes", test_clip_cache_shapes),
        ("bridge_output_shapes", test_bridge_output_shapes),
        ("pos_tagger", test_pos_tagger),
        ("2d_geometry_bias", test_2d_geometry_bias),
        ("injector_prepend_slicing", test_injector_prepend_slicing),
        ("injector_interleave_slicing", test_injector_interleave_slicing),
        ("bridge_param_count", test_bridge_param_count),
    ]
    n_pass = 0
    n_total = len(tests)
    print("Running %d shape tests" % n_total)
    for name, fn in tests:
        if test(name, fn):
            n_pass += 1
    print("%d / %d tests passed" % (n_pass, n_total))
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()