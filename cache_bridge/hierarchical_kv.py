# -*- coding: utf-8 -*-
"""Hierarchical 4-layer key-value cache for the visual encoder.

Implements the four-layer decomposition described in the manual:
    basic (pixel textures, local edges),
    semantic-object (object classes, instance boxes),
    spatial-relation (object-to-object positional relations),
    attribute (color, material, state).

Each layer is a small (key, value) tensor pair that the bridge adapter
will transform into a language-model-compatible representation.
"""

import numpy as np


class LayerKV(object):
    """One hierarchical layer of (K, V) tokens.

    Attributes
    ----------
    keys   : ndarray, shape (n_tokens, dim)
    values : ndarray, shape (n_tokens, dim)
    coords : ndarray, shape (n_tokens, coord_dim) -- spatial coordinates
             used by the structure-preserving attention unit.
    meta   : list of strings, human readable label per token
    """

    __slots__ = ("keys", "values", "coords", "meta", "layer_name")

    def __init__(self, keys, values, coords, meta, layer_name):
        self.keys = keys
        self.values = values
        self.coords = coords
        self.meta = meta
        self.layer_name = layer_name

    def __len__(self):
        return self.keys.shape[0]

    def to_dict(self):
        return {
            "keys": self.keys,
            "values": self.values,
            "coords": self.coords,
            "meta": self.meta,
            "layer_name": self.layer_name,
        }


class HierarchicalKVCache(object):
    """Container holding the four LayerKV instances.

    The four layers are produced by the visual encoder, then handed to
    the bridge adapter which projects them into the language model's
    embedding space and fuses them with a structure-preserving attention
    that respects the relative coordinates of every token.
    """

    LAYER_NAMES = ("basic", "semantic_object", "spatial_relation", "attribute")

    def __init__(self, basic, semantic_object, spatial_relation, attribute):
        self.layers = {
            "basic": basic,
            "semantic_object": semantic_object,
            "spatial_relation": spatial_relation,
            "attribute": attribute,
        }

    @classmethod
    def from_layers(cls, layers):
        """Build from a dict mapping layer_name -> LayerKV."""
        for name in cls.LAYER_NAMES:
            if name not in layers:
                raise KeyError("missing layer %s" % name)
        return cls(
            layers["basic"],
            layers["semantic_object"],
            layers["semantic_relation" if "semantic_relation" in layers else "spatial_relation"],
            layers["attribute"],
        )

    def total_tokens(self):
        return sum(len(layer) for layer in self.layers.values())

    def get(self, name):
        return self.layers[name]

    def concat_tokens(self):
        """Concatenate all four layers along the token axis.

        Returns
        -------
        K, V : ndarray, shape (total_tokens, dim)
        coords : ndarray, shape (total_tokens, 2)
        layer_id : ndarray, shape (total_tokens,) of int8 in {0,1,2,3}
        meta : list of strings
        """
        ks, vs, cs, ls, ms = [], [], [], [], []
        for i, name in enumerate(self.LAYER_NAMES):
            layer = self.layers[name]
            n = len(layer)
            ks.append(layer.keys)
            vs.append(layer.values)
            cs.append(layer.coords)
            ls.append(np.full(n, i, dtype=np.int8))
            ms.extend(layer.meta)
        return (
            np.concatenate(ks, axis=0),
            np.concatenate(vs, axis=0),
            np.concatenate(cs, axis=0),
            np.concatenate(ls, axis=0),
            ms,
        )

    def select_layers(self, layer_mask):
        """Build a new cache containing only the layers whose bit is set.

        layer_mask: iterable of length 4 with 0/1 values, in the order
                    (basic, semantic_object, spatial_relation, attribute)
        """
        new_layers = {}
        for i, name in enumerate(self.LAYER_NAMES):
            if layer_mask[i]:
                new_layers[name] = self.layers[name]
        # The downstream code always expects four layers, so we keep
        # an empty placeholder (zero tokens) for the missing ones.
        empty = LayerKV(
            keys=np.zeros((0, self.layers[name].keys.shape[1]), dtype=np.float32),
            values=np.zeros((0, self.layers[name].values.shape[1]), dtype=np.float32),
            coords=np.zeros((0, self.layers[name].coords.shape[1]), dtype=np.float32),
            meta=[],
            layer_name=name,
        )
        for name in self.LAYER_NAMES:
            if name not in new_layers:
                new_layers[name] = empty
        return HierarchicalKVCache(
            new_layers["basic"],
            new_layers["semantic_object"],
            new_layers["spatial_relation"],
            new_layers["attribute"],
        )