# -*- coding: utf-8 -*-
"""Small GPT-style language model with optional cross-attention to a
visual cache produced by the bridge adapter.

The model has its own token embedding table, a stack of causal
self-attention blocks, and a cross-attention block at every layer
that takes the visual K, V produced by the bridge adapter as an
additional source of context.  The output is the standard next-token
logits.

The NumPy implementation is intentionally small and well-commented so
that the data flow matches the architecture in the patent.
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


class _CausalSelfAttention(object):
    def __init__(self, dim, n_heads, max_seq_len, seed=0):
        assert dim % n_heads == 0
        self.dim = dim
        self.h = n_heads
        self.dh = dim // n_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wk = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wv = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wo = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        # Causal mask buffer.
        mask = np.triu(np.ones((max_seq_len, max_seq_len), dtype=np.float32) * -1e9, k=1)
        self.causal_mask = mask

    def __call__(self, x):
        # x: (B, T, dim)
        B, T, _ = x.shape
        q = np.dot(x, self.Wq)
        k = np.dot(x, self.Wk)
        v = np.dot(x, self.Wv)
        q = q.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        attn = np.einsum('bnqd,bnkd->bnqk', q, k) / np.sqrt(self.dh)
        attn = attn + self.causal_mask[None, None, :T, :T]
        p = _softmax(attn, axis=-1)
        out = np.einsum('bnqk,bnkd->bnqd', p, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return np.dot(out, self.Wo)


class _CrossAttention(object):
    """Cross-attention from text positions to the visual cache.

    The visual cache provides K and V.  Queries come from the text
    representation.  The visual K, V are concatenated to the per-layer
    K, V of the text in a way that lets the model attend to both with
    a single set of attention scores.
    """

    def __init__(self, dim, n_heads, seed=0):
        assert dim % n_heads == 0
        self.dim = dim
        self.h = n_heads
        self.dh = dim // n_heads
        rng = np.random.RandomState(seed)
        self.Wq = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wk_vis = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wv_vis = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.Wo = rng.randn(dim, dim).astype(np.float32) * (1.0 / np.sqrt(dim))
        # Per-injection-position scaling (slightly different behaviour
        # when the visual cache is injected at a noun vs a preposition
        # position).
        self.position_scale = rng.randn(3).astype(np.float32) * 0.05 + 1.0

    def __call__(self, text_x, vis_K, vis_V, position_id):
        # text_x: (B, T, dim), vis_K/vis_V: (B, N, dim) (or (N, dim))
        B, T, _ = text_x.shape
        if vis_K.ndim == 2:
            vis_K = np.tile(vis_K[None], (B,) + (1,) * vis_K.ndim)
            vis_V = np.tile(vis_V[None], (B,) + (1,) * vis_V.ndim)
        N = vis_K.shape[1]
        q = np.dot(text_x, self.Wq)
        k_vis = np.dot(vis_K, self.Wk_vis)
        v_vis = np.dot(vis_V, self.Wv_vis)
        q = q.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        k_vis = k_vis.reshape(B, N, self.h, self.dh).transpose(0, 2, 1, 3)
        v_vis = v_vis.reshape(B, N, self.h, self.dh).transpose(0, 2, 1, 3)
        attn = np.einsum('bnqd,bnkd->bnqk', q, k_vis) / np.sqrt(self.dh)
        # Per-position scaling.  We pick a single scalar per text
        # position based on the supplied position_id, then broadcast.
        scale = self.position_scale[position_id]  # (B, T)
        attn = attn * scale[:, None, :, None]
        p = _softmax(attn, axis=-1)
        out = np.einsum('bnqk,bnkd->bnqd', p, v_vis)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.dim)
        return np.dot(out, self.Wo)


class _MLP(object):
    def __init__(self, dim, hidden, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(dim, hidden).astype(np.float32) * (1.0 / np.sqrt(dim))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, dim).astype(np.float32) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(dim, dtype=np.float32)

    def __call__(self, x):
        return np.dot(_gelu(np.dot(x, self.W1) + self.b1), self.W2) + self.b2


class _DecoderBlock(object):
    """One transformer block with causal self-attention + cross-attn."""

    def __init__(self, dim, n_heads, ffn_hidden, seed=0):
        self.ln1_g = np.ones(dim, dtype=np.float32)
        self.ln1_b = np.zeros(dim, dtype=np.float32)
        self.self_attn = _CausalSelfAttention(dim, n_heads, 128, seed)
        self.ln_cross_g = np.ones(dim, dtype=np.float32)
        self.ln_cross_b = np.zeros(dim, dtype=np.float32)
        self.cross_attn = _CrossAttention(dim, n_heads, seed + 1)
        self.ln2_g = np.ones(dim, dtype=np.float32)
        self.ln2_b = np.zeros(dim, dtype=np.float32)
        self.mlp = _MLP(dim, ffn_hidden, seed + 2)

    def __call__(self, x, vis_K, vis_V, position_id):
        x = x + self.self_attn(_layer_norm(x, self.ln1_g, self.ln1_b))
        if vis_K is not None and vis_K.shape[0] > 0:
            x = x + self.cross_attn(
                _layer_norm(x, self.ln_cross_g, self.ln_cross_b),
                vis_K,
                vis_V,
                position_id,
            )
        x = x + self.mlp(_layer_norm(x, self.ln2_g, self.ln2_b))
        return x


class LanguageModel(object):
    """Tiny GPT-style language model with visual cache cross-attention."""

    def __init__(self, config, seed=0):
        self.config = config
        rng = np.random.RandomState(seed)
        self.token_emb = rng.randn(config.vocab_size, config.language_dim).astype(np.float32) * 0.05
        self.pos_emb = rng.randn(config.max_seq_len, config.language_dim).astype(np.float32) * 0.05
        self.blocks = [
            _DecoderBlock(config.language_dim, config.num_heads, config.ffn_hidden, seed=seed + 10 * i)
            for i in range(config.num_decoder_layers)
        ]
        self.final_ln_g = np.ones(config.language_dim, dtype=np.float32)
        self.final_ln_b = np.zeros(config.language_dim, dtype=np.float32)
        self.lm_head = rng.randn(config.language_dim, config.vocab_size).astype(np.float32) * 0.05

    def embed(self, token_ids):
        # token_ids: (B, T) integer
        x = self.token_emb[token_ids]
        T = x.shape[1]
        x = x + self.pos_emb[:T][None]
        return x

    def forward(self, token_ids, vis_K=None, vis_V=None, position_id=None):
        x = self.embed(token_ids)
        if position_id is None:
            position_id = np.zeros(x.shape[:2], dtype=np.int64)
        for block in self.blocks:
            x = block(x, vis_K, vis_V, position_id)
        x = _layer_norm(x, self.final_ln_g, self.final_ln_b)
        logits = np.dot(x, self.lm_head)
        return logits

    def greedy_generate(self, prompt_ids, vis_K, vis_V, position_id, max_new_tokens=20):
        ids = list(prompt_ids)
        prompt_pos = position_id[0] if position_id is not None else np.zeros(len(ids), dtype=np.int64)
        for _ in range(max_new_tokens):
            x = np.array([ids[-self.config.max_seq_len:]], dtype=np.int64)
            T = x.shape[1]
            pos = np.zeros(T, dtype=np.int64)
            n_prompt = min(len(prompt_pos), T)
            pos[:n_prompt] = prompt_pos[:n_prompt]
            logits = self.forward(x, vis_K, vis_V, pos[None])
            next_id = int(np.argmax(logits[0, -1]))
            ids.append(next_id)
        return np.array(ids, dtype=np.int64)