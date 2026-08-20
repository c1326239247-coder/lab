# -*- coding: utf-8 -*-
"""Dynamic routing controller.

The router receives a query embedding and decides:
  * which of the four cache layers should be activated,
  * the per-position injection strategy (object -> noun position,
    relation -> preposition position, attribute -> adjective position),
  * the gate bias to apply to the bridge adapter.

The implementation is a small NumPy MLP that maps a query embedding to
three outputs:
  layer_logits      : (n_layers,) raw scores for the four cache layers
  position_weights  : (3,) the soft weights for the three
                      position-types (noun, preposition, adjective)
  gate_bias         : scalar, shifts the gate of the bridge adapter

The activation of layers is the binary pattern that selects the four
bits (basic, semantic_object, spatial_relation, attribute).  The
position_weights are used by the language model to scale the
cross-attention per text position.
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class DynamicRouter(object):
    QUERY_TYPES = ("detailed_description", "spatial_reasoning", "subjective")
    # Layer activation pattern (basic, semantic_object, spatial_relation, attribute)
    DEFAULT_PATTERNS = {
        "detailed_description": np.array([1, 1, 1, 1], dtype=np.float32),
        "spatial_reasoning": np.array([0, 1, 1, 0], dtype=np.float32),
        "subjective": np.array([0, 0, 0, 1], dtype=np.float32),
    }

    def __init__(self, query_dim, n_layers=4, seed=0):
        rng = np.random.RandomState(seed)
        self.query_dim = query_dim
        self.n_layers = n_layers
        self.W1 = rng.randn(query_dim, query_dim).astype(np.float32) * 0.2
        self.b1 = np.zeros(query_dim, dtype=np.float32)
        self.W_layers = rng.randn(query_dim, n_layers).astype(np.float32) * 0.1
        self.b_layers = np.zeros(n_layers, dtype=np.float32)
        self.W_pos = rng.randn(query_dim, 3).astype(np.float32) * 0.1
        self.b_pos = np.zeros(3, dtype=np.float32)
        self.W_gate = rng.randn(query_dim, 1).astype(np.float32) * 0.1
        self.b_gate = np.zeros(1, dtype=np.float32)

    def __call__(self, query_embedding):
        # query_embedding: (dim,) or (B, dim)
        single = (query_embedding.ndim == 1)
        if single:
            q = query_embedding[None]
        h = np.tanh(np.dot(q, self.W1) + self.b1)
        layer_logits = np.dot(h, self.W_layers) + self.b_layers
        layer_probs = _sigmoid(layer_logits)
        # Threshold to a binary pattern, with a hysteresis-style rule:
        # the top-2 always get activated, the rest follow the probability.
        layer_mask = np.zeros_like(layer_probs)
        for b in range(layer_probs.shape[0]):
            top2 = np.argsort(-layer_probs[b])[:2]
            layer_mask[b, top2] = 1
            for i in range(layer_probs.shape[1]):
                if i not in top2 and layer_probs[b, i] > 0.5:
                    layer_mask[b, i] = 1
        pos_logits = np.dot(h, self.W_pos) + self.b_pos
        pos_weights = _softmax(pos_logits, axis=-1)
        gate_bias = _sigmoid(np.dot(h, self.W_gate) + self.b_gate)
        if single:
            return layer_mask[0], pos_weights[0], gate_bias[0, 0]
        return layer_mask, pos_weights, gate_bias[:, 0]

    def classify_query(self, query_tokens, token_emb):
        """Map a query string of token ids to a (layer_mask, pos_weights, gate_bias)."""
        if len(query_tokens) == 0:
            emb = np.zeros(self.query_dim, dtype=np.float32)
        else:
            emb = token_emb[query_tokens].mean(axis=0)
        return self(emb)

    def default_pattern(self, query_type):
        return self.DEFAULT_PATTERNS[query_type].copy()