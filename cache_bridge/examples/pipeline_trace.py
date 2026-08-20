# -*- coding: utf-8 -*-
"""Example: end-to-end pipeline trace on a single example.

This is the script the user can run to verify the whole pipeline
works.  It produces a small JSON file describing the data flow at
each stage so a reader can inspect exactly what the model is doing.
"""

import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARENT = os.path.dirname(ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from cache_bridge.config import default_config
from cache_bridge.data import SyntheticMultimodalDataset
from cache_bridge.pipeline import CacheBridgePipeline


def _to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def main():
    cfg = default_config()
    print("Config:", cfg)
    pipeline = CacheBridgePipeline(cfg, seed=0)
    dataset = SyntheticMultimodalDataset(2, cfg, seed=0)
    ex = dataset[0]
    print("Example object :", ex["object_label"])
    print("Query tokens   :", ex["query_token_ids"].tolist())
    print("Position ids   :", ex["position_id"].tolist())

    cache = pipeline.encode_image(ex["image"])
    layer_sizes = {
        name: {"keys": cache.get(name).keys.shape,
               "coords": cache.get(name).coords.shape}
        for name in cache.layers.keys()
    }
    print("Cache layer sizes:", layer_sizes)

    q_emb = pipeline.lm.token_emb[ex["query_token_ids"]].mean(axis=0)
    layer_mask, pos_weights, gate_bias = pipeline.router(q_emb)
    print("Router output:")
    print("  layer mask :", layer_mask.tolist())
    print("  pos weights:", pos_weights.tolist())
    print("  gate bias  :", float(gate_bias))

    text_feats = pipeline.lm.embed(ex["query_token_ids"][None])[0]
    K, V = pipeline.adapter(cache, text_feats, layer_mask)
    print("Bridge output K, V shape:", K.shape, V.shape)

    logits, mask, pos, gate = pipeline.forward(
        ex["image"], ex["query_token_ids"], ex["position_id"]
    )
    print("Logits shape:", logits.shape)

    out = pipeline.generate(
        ex["image"], ex["query_token_ids"], ex["position_id"], max_new_tokens=10
    )
    out_text = "".join(chr(int(i)) for i in out if 0 <= int(i) < 256)
    print("Greedy output :", out_text)

    trace = {
        "config": _to_python(cfg._asdict()),
        "example": {
            "object": ex["object_label"],
            "query": ex["query_token_ids"].tolist(),
            "response": ex["response_token_ids"].tolist(),
            "position_id": ex["position_id"].tolist(),
        },
        "cache_layer_sizes": _to_python(layer_sizes),
        "router": {
            "layer_mask": layer_mask.tolist(),
            "pos_weights": pos_weights.tolist(),
            "gate_bias": float(gate_bias),
        },
        "bridge": {
            "K_shape": list(K.shape),
            "V_shape": list(V.shape),
        },
        "lm": {
            "logits_shape": list(logits.shape),
        },
        "generation": {
            "output_ids": out.tolist(),
            "output_text": out_text,
        },
    }
    out_path = os.path.join(ROOT, "trace.json")
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print("Trace saved to:", out_path)


if __name__ == "__main__":
    main()