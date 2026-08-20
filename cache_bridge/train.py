# -*- coding: utf-8 -*-
"""Entry point for the cache-bridging system.

Usage:
    py -m cache_bridge.train            # short smoke test
    py -m cache_bridge.train --full     # longer training run
    py -m cache_bridge.train --demo     # skip training, just run a forward pass
"""

import sys
import time
import argparse
import numpy as np

from .config import default_config, small_demo_config, tiny_smoke_config
from .data import SyntheticMultimodalDataset
from .pipeline import CacheBridgePipeline
from .training import Trainer
from .dynamic_router import DynamicRouter


def _bytes_to_text(ids):
    return "".join(chr(int(i)) for i in ids if 0 <= int(i) < 256)


def demo(config=None, n_examples=8, seed=0):
    cfg = config or default_config()
    print("Using config:")
    print(" ", cfg)
    print()
    pipeline = CacheBridgePipeline(cfg, seed=seed)
    print("Pipeline:", pipeline.summary())
    dataset = SyntheticMultimodalDataset(n_examples, cfg, seed=seed)
    print("Dataset size: %d" % len(dataset))
    for i, ex in enumerate(dataset):
        print()
        print("--- Example %d ---" % i)
        print("  object  :", ex["object_label"])
        print("  type    :", ex["query_type"])
        print("  query   :", _bytes_to_text(ex["query_token_ids"]))
        print("  response:", _bytes_to_text(ex["response_token_ids"]))
        t0 = time.time()
        logits, layer_mask, pos_weights, gate_bias = pipeline.forward(
            ex["image"], ex["query_token_ids"], ex["position_id"]
        )
        dt = time.time() - t0
        print("  layer mask (basic, obj, rel, attr):", layer_mask)
        print("  pos weights (noun, prep, adj)    :", np.round(pos_weights, 3))
        print("  gate bias                         :", round(float(gate_bias), 3))
        print("  forward time                      : %.3f s" % dt)
        print("  logits shape                      :", logits.shape)
        pred = np.argmax(logits[0], axis=-1)
        print("  greedy argmax                     :", _bytes_to_text(pred))


def train(config=None, stage1=10, stage2=10, stage3=10, n_examples=12, seed=0):
    cfg = config or default_config()
    pipeline = CacheBridgePipeline(cfg, seed=seed)
    dataset = SyntheticMultimodalDataset(n_examples, cfg, seed=seed)
    print("Pipeline:", pipeline.summary())
    print("Dataset size: %d" % len(dataset))
    trainer = Trainer(pipeline, dataset, cfg)
    t0 = time.time()
    trainer.run(stage1_iters=stage1, stage2_iters=stage2, stage3_iters=stage3)
    print("Total training time: %.1f s" % (time.time() - t0))
    # Run a quick generation demo.
    ex = dataset[0]
    print()
    print("--- Generation demo ---")
    print("Query   :", _bytes_to_text(ex["query_token_ids"]))
    out = pipeline.generate(
        ex["image"],
        ex["query_token_ids"],
        ex["position_id"],
        max_new_tokens=15,
    )
    print("Output  :", _bytes_to_text(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cache-bridging cross-modal system")
    parser.add_argument("--demo", action="store_true", help="just run a forward pass")
    parser.add_argument("--full", action="store_true", help="use the larger config")
    parser.add_argument("--stage1", type=int, default=10)
    parser.add_argument("--stage2", type=int, default=10)
    parser.add_argument("--stage3", type=int, default=10)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny", action="store_true", help="use the tiny smoke config")
    args = parser.parse_args(argv)
    if args.tiny:
        cfg = tiny_smoke_config()
    elif args.full:
        cfg = small_demo_config()
    else:
        cfg = default_config()
    if args.demo:
        demo(cfg, n_examples=args.examples, seed=args.seed)
    else:
        train(cfg, stage1=args.stage1, stage2=args.stage2, stage3=args.stage3,
              n_examples=args.examples, seed=args.seed)


if __name__ == "__main__":
    main()