# -*- coding: utf-8 -*-
"""Bridge adapter module.

Maps the visual encoder's hierarchical KV cache into a representation
the language model can consume.  Three sub-modules, in order:

  1. DimensionProjection      - lifts the vision tokens to the
     language model embedding size and creates a per-layer pair of
     (K_bridge, V_bridge) tensors.
  2. StructurePreservingAttn  - injects relative position encodings so
     that the spatial relations between objects are preserved through
     the projection.
  3. AdaptiveGating           - learns a per-token gate that mixes the
     projected visual features with the language model's own
     embeddings, preventing the visual cache from overwhelming the
     generation when the query does not actually need it.

The output is a (K, V) pair the language model can use as cross-attention
input.  The whole adapter is a NumPy-only neural module.
"""

import numpy as np


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _relative_pos_bias(queries, keys, coords_q, coords_k, n_heads, gamma, beta):
    """Compute a multi-head attention bias from the spatial coordinates.

    The bias is a smooth function of the relative distance between each
    (query, key) pair of tokens.  This is the core of the
    "structure-preserving" attention: the relative geometry survives
    the projection, even though the bridge now lives in a completely
    different feature space than the encoder that produced the
    coordinates.
    """
    # coords: (T, 2).  Pairwise Euclidean distance.
    diff = coords_q[:, None, :] - coords_k[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)  # (Tq, Tk)
    # A small MLP per head produces a scalar bias.
    Tq, Tk = dist.shape
    dh = gamma.shape[0] // n_heads
    bias = np.zeros((Tq, Tk, n_heads), dtype=np.float32)
    for h in range(n_heads):
        g = gamma[h * dh:(h + 1) * dh]
        b = beta[h * dh:(h + 1) * dh]
        h_in = g[0] * dist + b[0]
        bias[:, :, h] = h_in
    return bias.transpose(2, 0, 1)  # (n_heads, Tq, Tk)


class DimensionProjection(object):
    """Lifts the visual cache to the language model dimension."""

    def __init__(self, in_dim, out_dim, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(in_dim, out_dim).astype(np.float32) * (1.0 / np.sqrt(in_dim))
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.ln_g = np.ones(out_dim, dtype=np.float32)
        self.ln_b = np.zeros(out_dim, dtype=np.float32)

    def __call__(self, x):
        h = np.dot(x, self.W) + self.b
        return _layer_norm(h, self.ln_g, self.ln_b)


class StructurePreservingAttention(object):
    """Self-attention with a relative position bias.

    Operates on the per-layer visual cache tokens.  We need three
    tensors:
        x        : (T, dim) the visual tokens
        coords   : (T, 2)   the spatial coordinates of each token
        layer_id : (T,)     integer in {0,1,2,3} naming the layer

    The output is the same shape as `x` and contains the visual tokens
    refined by attending to each other using both content and geometry.
    """

    def __init__(self, dim, n_heads, seed=0):
        assert dim % n_heads == 0
        self.dim = dim
        self.h = n_heads
        self.dh = dim // n_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wk = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wv = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wo = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.bias_gamma = rng.randn(dim).astype(np.float32) * 0.1
        self.bias_beta = np.zeros(dim, dtype=np.float32)
        # Per-layer additive bias (so basic/object/relation/attribute
        # tokens have slightly different attention patterns).
        self.layer_bias = rng.randn(4, n_heads).astype(np.float32) * 0.05

    def __call__(self, x, coords, layer_id):
        T = x.shape[0]
        q = np.dot(x, self.Wq).reshape(T, self.h, self.dh).transpose(1, 0, 2)
        k = np.dot(x, self.Wk).reshape(T, self.h, self.dh).transpose(1, 0, 2)
        v = np.dot(x, self.Wv).reshape(T, self.h, self.dh).transpose(1, 0, 2)
        attn = np.einsum('hqd,hkd->hqk', q, k) / np.sqrt(self.dh)
        # Geometry bias per head.
        geo_bias = _relative_pos_bias(coords, coords, coords, coords, self.h, self.bias_gamma, self.bias_beta)
        # layer_id bias
        layer_bias = self.layer_bias[layer_id].T  # (h, T)
        attn = attn + geo_bias + layer_bias[:, None, :]
        p = _softmax(attn, axis=-1)
        out = np.einsum('hqk,hkd->hqd', p, v)
        out = out.transpose(1, 0, 2).reshape(T, self.dim)
        return np.dot(out, self.Wo)


class AdaptiveGating(object):
    """Per-token gate that mixes the visual features with the language
    model's text features.

    The gate g is in [0, 1]^T and is produced by a small linear layer
    over the concatenation of the visual feature and the language
    model's own text feature for that position.  During training the
    gate learns to silence visual tokens that are not useful for the
    current query.
    """

    def __init__(self, dim, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(2 * dim, dim).astype(np.float32) * 0.1
        self.b = np.zeros(dim, dtype=np.float32)
        self.ln_g = np.ones(dim, dtype=np.float32)
        self.ln_b = np.zeros(dim, dtype=np.float32)

    def gate(self, visual_feat, text_feat):
        # visual_feat, text_feat: (T, dim)
        T, dim = visual_feat.shape
        x = np.concatenate([visual_feat, text_feat], axis=-1)
        h = _gelu(np.dot(x, self.W) + self.b)
        h = _layer_norm(h, self.ln_g, self.ln_b)
        # squash per-dim scores to a scalar gate via mean.
        return _sigmoid(h.mean(axis=-1, keepdims=True))

    def __call__(self, visual_feat, text_feat):
        g = self.gate(visual_feat, text_feat)  # (T, 1)
        return g * visual_feat + (1.0 - g) * text_feat


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BridgeAdapter(object):
    """Combines the three sub-modules into a single callable.

    Inputs
    ------
    cache   : HierarchicalKVCache (output of the visual encoder)
    text_feats : (T_text, dim) the language model's own embeddings for
                 each text position where the visual cache should be
                 injected.  When a position is not part of the injection
                 (e.g. punctuation), the matching row in text_feats can
                 be the zero vector and the gate will close.

    Output
    ------
    (K, V)  : (T_visual, dim) the cache ready to be used by the
              language model.  Returns K, V of the same shape.
    """

    def __init__(self, config, seed=0):
        self.config = config
        self.proj = DimensionProjection(config.vision_dim, config.language_dim, seed=seed)
        self.attn = StructurePreservingAttention(config.language_dim, config.num_heads, seed=seed + 1)
        self.gate = AdaptiveGating(config.language_dim, seed=seed + 2)
        self.layer_norm_g = np.ones(config.language_dim, dtype=np.float32)
        self.layer_norm_b = np.zeros(config.language_dim, dtype=np.float32)

    def _concat(self, cache, layer_mask):
        """Concatenate the four layers, respecting layer_mask (0/1)."""
        ks, vs, cs, ls = [], [], [], []
        for i, name in enumerate(HierarchicalKVCache_LN):
            if not layer_mask[i]:
                continue
            layer = cache.get(name)
            n = len(layer)
            if n == 0:
                continue
            ks.append(layer.keys)
            vs.append(layer.values)
            cs.append(layer.coords)
            ls.append(np.full(n, i, dtype=np.int8))
        if not ks:
            return (
                np.zeros((0, self.config.language_dim), dtype=np.float32),
                np.zeros((0, self.config.language_dim), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int8),
            )
        K_v = np.concatenate(ks, axis=0)
        V_v = np.concatenate(vs, axis=0)
        coords_c = np.concatenate(cs, axis=0)
        ls_c = np.concatenate(ls, axis=0)
        return K_v, V_v, coords_c, ls_c

    def __call__(self, cache, text_feats, layer_mask):
        K_v, V_v, coords, layer_id = self._concat(cache, layer_mask)
        if K_v.shape[0] == 0:
            return K_v, V_v
        # 1) project into language space
        K = self.proj(K_v)
        V = self.proj(V_v)
        # 2) structure-preserving self-attention
        K = K + self.attn(K, coords, layer_id)
        V = V + self.attn(V, coords, layer_id)
        K = _layer_norm(K, self.layer_norm_g, self.layer_norm_b)
        V = _layer_norm(V, self.layer_norm_g, self.layer_norm_b)
        # 3) adaptive gating.  We need text_feats of the right size; if
        # they are not, broadcast the mean over text.
        T = K.shape[0]
        if text_feats.shape[0] != T:
            mean_text = text_feats.mean(axis=0, keepdims=True)
            text_feats = np.tile(mean_text, (T, 1))
        K = self.gate(K, text_feats)
        V = self.gate(V, text_feats)
        return K, V


# Local re-export to keep the file self-contained.
HierarchicalKVCache_LN = ("basic", "semantic_object", "spatial_relation", "attribute")