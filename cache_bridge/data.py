# -*- coding: utf-8 -*-
"""Synthetic multimodal data for the cache-bridging system.

This module is deliberately minimal.  It does not depend on any
external dataset or tokenizer.  Each "example" contains:

  image             : (H, W, 3) uint8 array, a synthetic image
  query_token_ids   : 1-D int array, the question/prompt
  response_token_ids: 1-D int array, the target response
  position_id       : 1-D int array in {0, 1, 2}, the per-text-position
                      injection type (0=noun, 1=preposition, 2=adjective)
  query_type        : string, one of the router's three classes
  object_label      : a short string naming the dominant object in the
                      image (used by the synthetic generator)

The token vocabulary is byte-level (vocab_size = 256).  Each "text"
position is just a single byte 0-255.  This keeps the training loop
simple and avoids a real tokenizer while still exercising every part
of the model end-to-end.
"""

import numpy as np


def _make_synthetic_image(config, object_label, seed):
    """Render a small synthetic image based on the object label.

    The image is a hand-crafted grid where the position of a coloured
    patch encodes the "object" and the colour encodes the "attribute".
    A 32x32 image at patch_size=8 has 4x4 = 16 patches; the first row
    encodes the attribute and the first column encodes the object.
    """
    rng = np.random.RandomState(seed)
    H = config.image_size
    W = config.image_size
    img = np.zeros((H, W, 3), dtype=np.float32)
    ps = config.patch_size
    n = H // ps
    # Object hash: position of the highlighted cell.
    obj_hash = (sum(ord(c) for c in object_label) % (n * n))
    obj_y, obj_x = divmod(obj_hash, n)
    # Attribute hash: red/green/blue dominant colour.
    attr_hash = (sum(ord(c) for c in object_label[::-1]) % 3)
    for i in range(n):
        for j in range(n):
            color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            if i == obj_y or j == obj_x:
                color[attr_hash] = 1.0
            color += rng.normal(0.0, 0.05, size=3).astype(np.float32)
            color = np.clip(color, 0.0, 1.0)
            img[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps] = color
    return (img * 255.0).astype(np.uint8)


def _tokenize_bytes(text):
    if isinstance(text, str):
        text = text.encode("utf-8")
    elif isinstance(text, unicode):
        text = text.encode("utf-8")
    return np.array([ord(c) for c in text if ord(c) < 256], dtype=np.int64)


# Per-position injection type assignment.  This is a simple proxy for
# the position-aware injection described in the manual: the first noun
# in the prompt is tagged 0, the preposition is tagged 1, the
# adjective is tagged 2, and everything else is tagged 0.
def _tag_positions(token_ids):
    tags = np.zeros(len(token_ids), dtype=np.int64)
    # Simple heuristic: any token whose ASCII code is in [97, 122] is
    # considered a noun or adjective; 32 -> space, 110 -> 'n', etc.
    # We mark a few hand-picked positions so the language model's
    # cross-attention has something to differentiate.
    for i, t in enumerate(token_ids):
        c = int(t)
        if 97 <= c <= 122:  # lowercase letter -> noun slot
            tags[i] = 0
        elif c in (32,):  # space -> preposition slot
            tags[i] = 1
        elif 65 <= c <= 90:  # uppercase -> adjective slot
            tags[i] = 2
    return tags


_QUERY_TEMPLATES = {
    "detailed_description": [
        b"describe the {obj} in detail",
        b"what does the {obj} look like",
        b"give a full picture of the {obj}",
    ],
    "spatial_reasoning": [
        b"where is the {obj} located",
        b"is the {obj} on the left or right",
        b"how is the {obj} positioned",
    ],
    "subjective": [
        b"is the {obj} beautiful",
        b"what feeling does the {obj} give you",
        b"would you like the {obj}",
    ],
}

_RESPONSE_TEMPLATES = {
    "detailed_description": [
        b"a bright {obj} at the centre of the image",
        b"the {obj} sits on a neutral background",
        b"a vivid {obj} with sharp edges",
    ],
    "spatial_reasoning": [
        b"the {obj} is on the left half of the image",
        b"the {obj} appears in the upper right corner",
        b"the {obj} is centred horizontally",
    ],
    "subjective": [
        b"the {obj} feels calm and elegant",
        b"the {obj} has a warm and inviting presence",
        b"the {obj} looks impressive and bold",
    ],
}

_OBJECTS = [
    "chair", "table", "lamp", "phone", "book",
    "cup", "bottle", "dog", "cat", "tree",
    "flower", "car", "ball", "door", "window",
]


class SyntheticMultimodalDataset(object):
    """A small in-memory dataset with a fixed vocabulary."""

    def __init__(self, n_examples, config, seed=0):
        self.config = config
        self.examples = []
        rng = np.random.RandomState(seed)
        for i in range(n_examples):
            obj = _OBJECTS[rng.randint(0, len(_OBJECTS))]
            qtype = rng.choice(DynamicRouter_QUERY_TYPES)
            tmpl_q = _QUERY_TEMPLATES[qtype][rng.randint(0, len(_QUERY_TEMPLATES[qtype]))]
            tmpl_r = _RESPONSE_TEMPLATES[qtype][rng.randint(0, len(_RESPONSE_TEMPLATES[qtype]))]
            q = tmpl_q.replace(b"{obj}", obj.encode("utf-8"))
            r = tmpl_r.replace(b"{obj}", obj.encode("utf-8"))
            self.examples.append({
                "image": _make_synthetic_image(config, obj, seed=seed * 31 + i),
                "query_token_ids": _tokenize_bytes(q),
                "response_token_ids": _tokenize_bytes(r),
                "position_id": _tag_positions(_tokenize_bytes(q)),
                "query_type": qtype,
                "object_label": obj,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def vocab_size(self):
        return self.config.vocab_size


# Import the constant lazily to avoid a circular import.
DynamicRouter_QUERY_TYPES = (
    "detailed_description",
    "spatial_reasoning",
    "subjective",
)