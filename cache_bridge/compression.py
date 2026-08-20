# -*- coding: utf-8 -*-
"""Cache compression module.

Implements two optional compression steps mentioned in the manual:

  * Top-K selection -- keep the K most salient tokens per layer.
    The salience score is the L2 norm of the value vector.
  * Similarity-based merging -- when two tokens have a cosine
    similarity above `sim_threshold`, average their values and
    coordinates.  This reduces redundancy in the attribute and
    relation layers in particular.

The compression is applied independently to the (K, V) pair of each
layer, with the option to compress all four layers at once.
"""

import numpy as np


def topk_select(keys, values, coords, keep_ratio):
    if keep_ratio >= 1.0 or keys.shape[0] == 0:
        return keys, values, coords
    n = keys.shape[0]
    k = max(1, int(round(n * keep_ratio)))
    scores = np.linalg.norm(values, axis=-1)
    idx = np.argsort(-scores)[:k]
    return keys[idx], values[idx], coords[idx]


def cosine_merge(values, coords, sim_threshold):
    if values.shape[0] <= 1:
        return values, coords
    # Normalise
    norm = np.linalg.norm(values, axis=-1).reshape(-1, 1) + 1e-8
    n_values = values / norm
    sim = np.dot(n_values, n_values.T)
    n = sim.shape[0]
    keep = np.ones(n, dtype=bool)
    merged_into = np.arange(n, dtype=np.int64)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if sim[i, j] > sim_threshold:
                values[j] = (values[i] + values[j]) / 2.0
                coords[j] = (coords[i] + coords[j]) / 2.0
                keep[i] = False
                merged_into[i] = j
                break
    return values[keep], coords[keep]


def compress_layer(layer, keep_ratio, sim_threshold, do_topk=True, do_merge=True):
    keys, values, coords = layer.keys, layer.values, layer.coords
    if do_merge:
        keys, values, coords = _merge_kv(keys, values, coords, sim_threshold)
    if do_topk:
        keys, values, coords = topk_select(keys, values, coords, keep_ratio)
    return keys, values, coords

def _merge_kv(keys, values, coords, sim_threshold):
    if values.shape[0] <= 1:
        return keys, values, coords
    norm = np.linalg.norm(values, axis=-1).reshape(-1, 1) + 1e-8
    n_values = values / norm
    sim = np.dot(n_values, n_values.T)
    n = sim.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if sim[i, j] > sim_threshold:
                values[j] = (values[i] + values[j]) / 2.0
                coords[j] = (coords[i] + coords[j]) / 2.0
                keys[j] = (keys[i] + keys[j]) / 2.0
                keep[i] = False
                break
    return keys[keep], values[keep], coords[keep]


class CompressionModule(object):
    """Top-level wrapper used by the main pipeline."""

    def __init__(self, keep_ratio=0.5, sim_threshold=0.95):
        self.keep_ratio = keep_ratio
        self.sim_threshold = sim_threshold

    def compress(self, cache):
        from .hierarchical_kv import HierarchicalKVCache, LayerKV
        new_layers = {}
        for name in HierarchicalKVCache.LAYER_NAMES:
            layer = cache.get(name)
            if len(layer) == 0:
                new_layers[name] = layer
                continue
            k, v, c = compress_layer(
                layer,
                self.keep_ratio,
                self.sim_threshold,
            )
            new_layers[name] = LayerKV(
                keys=k.astype(np.float32),
                values=v.astype(np.float32),
                coords=c.astype(np.float32),
                meta=layer.meta[:k.shape[0]],
                layer_name=name,
            )
        return HierarchicalKVCache(
            new_layers["basic"],
            new_layers["semantic_object"],
            new_layers["spatial_relation" if "semantic_relation" in new_layers else "spatial_relation"],
            new_layers["attribute"],
        )