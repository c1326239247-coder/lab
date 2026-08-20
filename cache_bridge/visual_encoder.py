# -*- coding: utf-8 -*-
"""Visual encoder that produces the hierarchical 4-layer KV cache.

A small Vision-Transformer-style encoder embeds the image into a flat
sequence of patch tokens.  Four lightweight heads then project the
patch tokens into four parallel layers:

  * basic  -- early-layer style features (pixel texture, edges)
  * semantic_object  -- object-level features pooled around the
    detected object regions
  * spatial_relation -- relational tokens whose coordinates encode
    the relative geometry of objects
  * attribute  -- global attribute features (colour/material/state)

The encoder is intentionally small and NumPy-only so the entire
pipeline can be exercised on a CPU.  The architectural details
(patch attention, region pooling, relational pooling) follow the
descriptions in the manual.
"""

import numpy as np
from .hierarchical_kv import HierarchicalKVCache, LayerKV


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


def _matmul(a, b):
    """np.dot wrapper used throughout the codebase (Python 2.7 safe)."""
    return np.dot(a, b)


class _SelfAttention(object):
    def __init__(self, dim, n_heads, seed=None):
        assert dim % n_heads == 0
        self.dim = dim
        self.h = n_heads
        self.dh = dim // n_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wk = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wv = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wo = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))

    def __call__(self, x, mask=None):
        # x: (N, T, dim)
        q = np.dot(x, self.Wq)
        k = np.dot(x, self.Wk)
        v = np.dot(x, self.Wv)
        N, T, _ = x.shape
        q = q.reshape(N, T, self.h, self.dh).transpose(0, 2, 1, 3)
        k = k.reshape(N, T, self.h, self.dh).transpose(0, 2, 1, 3)
        v = v.reshape(N, T, self.h, self.dh).transpose(0, 2, 1, 3)
        attn = np.einsum('nhqd,nhkd->nhqk', q, k) / np.sqrt(self.dh)
        if mask is not None:
            attn = attn + mask
        p = _softmax(attn, axis=-1)
        out = np.einsum('nhqk,nhkd->nhqd', p, v)
        out = out.transpose(0, 2, 1, 3).reshape(N, T, self.dim)
        return np.dot(out, self.Wo)


class _MLP(object):
    def __init__(self, dim, hidden, seed=None):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(dim, hidden).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, dim).astype(np.float32) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(dim, dtype=np.float32)

    def __call__(self, x):
        return np.dot(_gelu(np.dot(x, self.W1) + self.b1), self.W2) + self.b2


class _TransformerBlock(object):
    def __init__(self, dim, n_heads, ffn_hidden, seed=None):
        self.ln1_g = np.ones(dim, dtype=np.float32)
        self.ln1_b = np.zeros(dim, dtype=np.float32)
        self.attn = _SelfAttention(dim, n_heads, seed)
        self.ln2_g = np.ones(dim, dtype=np.float32)
        self.ln2_b = np.zeros(dim, dtype=np.float32)
        self.mlp = _MLP(dim, ffn_hidden, seed)

    def __call__(self, x, mask=None):
        x = x + self.attn(_layer_norm(x, self.ln1_g, self.ln1_b), mask=mask)
        x = x + self.mlp(_layer_norm(x, self.ln2_g, self.ln2_b))
        return x


class _PatchEncoder(object):
    """Embed the image into a sequence of patch tokens."""

    def __init__(self, config, seed=0):
        rng = np.random.RandomState(seed)
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.dim = config.vision_dim
        self.n_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = config.patch_size * config.patch_size * 3
        self.proj = rng.randn(patch_dim, config.vision_dim).astype(np.float32) * 0.05
        self.pos = rng.randn(self.n_patches, config.vision_dim).astype(np.float32) * 0.05
        self.cls = rng.randn(1, config.vision_dim).astype(np.float32) * 0.05

    def __call__(self, image):
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        H, W, _ = image.shape
        ps = self.patch_size
        ph = H // ps
        pw = W // ps
        patches = image[: ph * ps, : pw * ps].reshape(ph, ps, pw, ps, 3)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(ph * pw, ps * ps * 3)
        tokens = np.dot(patches, self.proj) + self.pos
        return np.concatenate([self.cls, tokens], axis=0)


class _LayerHead(object):
    """A small head that turns patch tokens into one cache layer.

    For each of the n_out output tokens we use a separate (W, query)
    pair, so the pooled outputs are inherently different even when the
    input patch tokens are similar.  This is what makes the
    compression module's cosine-merge step behave sensibly.
    """

    def __init__(self, n_out_tokens, dim, seed=0):
        rng = np.random.RandomState(seed)
        # Per-output-token W: (n_out, dim, dim)
        self.Ws = rng.randn(n_out_tokens, dim, dim).astype(np.float32) * 0.3
        self.b = np.zeros(dim, dtype=np.float32)
        # Per-output-token query vector: (n_out, dim)
        self.queries = rng.randn(n_out_tokens, dim).astype(np.float32) * 0.8
        # Per-output-token output projection
        self.Wo = rng.randn(dim, dim).astype(np.float32) * 0.3

    def _attn_pool(self, x, q):
        # x: (T, dim), q: (dim,) -> pooled (dim,)
        scores = np.dot(x, q)
        scores = scores / np.sqrt(x.shape[-1])
        p = _softmax(scores)
        return np.dot(p, x)

    def __call__(self, patch_tokens, coords):
        T = patch_tokens.shape[0]
        n_out = self.Ws.shape[0]
        pooled = np.zeros((n_out, patch_tokens.shape[1]), dtype=np.float32)
        for i in range(n_out):
            # Project patches with the per-output W.
            x = _gelu(np.dot(patch_tokens, self.Ws[i]) + self.b)
            pooled[i] = self._attn_pool(x, self.queries[i])
        pooled = np.dot(pooled, self.Wo)
        return pooled, coords


class VisualEncoder(object):
    """Vision Transformer that produces a HierarchicalKVCache."""

    def __init__(self, config, seed=0):
        self.config = config
        self.patch_encoder = _PatchEncoder(config, seed=seed)
        self.blocks = [
            _TransformerBlock(config.vision_dim, config.num_heads, config.ffn_hidden, seed=seed + i + 1)
            for i in range(2)
        ]
        self.basic_head = _LayerHead(config.num_basic_tokens, config.vision_dim, seed=seed + 10)
        self.object_head = _LayerHead(config.num_object_tokens, config.vision_dim, seed=seed + 20)
        self.relation_head = _LayerHead(config.num_relation_tokens, config.vision_dim, seed=seed + 30)
        self.attribute_head = _LayerHead(config.num_attribute_tokens, config.vision_dim, seed=seed + 40)
        self.basic_coords = self._make_coords(config.num_basic_tokens, seed + 1)
        self.object_coords = self._make_coords(config.num_object_tokens, seed + 2)
        self.relation_coords = self._make_coords(config.num_relation_tokens, seed + 3)
        self.attribute_coords = self._make_coords(config.num_attribute_tokens, seed + 4)
        self.object_meta = ["obj_%d" % i for i in range(config.num_object_tokens)]
        self.relation_meta = ["rel_%d" % i for i in range(config.num_relation_tokens)]
        self.attribute_meta = ["attr_%d" % i for i in range(config.num_attribute_tokens)]
        self.basic_meta = ["px_%d" % i for i in range(config.num_basic_tokens)]

    @staticmethod
    def _make_coords(n, seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)

    def encode(self, image):
        tokens = self.patch_encoder(image)
        for block in self.blocks:
            tokens = block(tokens[None])[0]
        patch_tokens = tokens[1:]

        basic_k, basic_coords = self.basic_head(patch_tokens, self.basic_coords)
        basic_v = basic_k
        obj_k, obj_coords = self.object_head(patch_tokens, self.object_coords)
        obj_v = obj_k
        rel_k, rel_coords = self.relation_head(patch_tokens, self.relation_coords)
        rel_v = rel_k
        attr_k, attr_coords = self.attribute_head(patch_tokens, self.attribute_coords)
        attr_v = attr_k

        return HierarchicalKVCache(
            basic=LayerKV(basic_k, basic_v, basic_coords, list(self.basic_meta), "basic"),
            semantic_object=LayerKV(obj_k, obj_v, obj_coords, list(self.object_meta), "semantic_object"),
            spatial_relation=LayerKV(rel_k, rel_v, rel_coords, list(self.relation_meta), "spatial_relation"),
            attribute=LayerKV(attr_k, attr_v, attr_coords, list(self.attribute_meta), "attribute"),
        )