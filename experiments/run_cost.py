# -*- coding: utf-8 -*-
"""Measure single-sample inference cost for each connector design.

Reported numbers are wall-clock latency for one greedy answer (6 new
tokens) and the peak allocated GPU memory of that forward pass, averaged
over `--repeats` questions after a warm-up.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import model as M
import run_vqa as R

common.setup_env()


def measure(system, batcher, rows, repeats, warmup=5):
    latencies = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i, row in enumerate(rows[:repeats + warmup]):
        px, ids, mask = batcher.eval_batch([row])
        torch.cuda.synchronize()
        start = time.time()
        system.generate(px, ids, mask, max_new_tokens=6)
        torch.cuda.synchronize()
        if i >= warmup:
            latencies.append((time.time() - start) * 1000.0)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    latencies.sort()
    return {
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1),
        "latency_ms_median": round(latencies[len(latencies) // 2], 1),
        "peak_mem_gb": round(peak, 2),
        "n_measured": len(latencies),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--n-visual", type=int, default=36)
    args = ap.parse_args()

    common.set_seed(0)
    clip, llm, tok, proc = R.load_backbones()
    cfg = M.BridgeCfg(clip, llm, n_visual=args.n_visual)
    batcher = R.Batcher(tok, proc)
    rows = common.load_gqa(args.repeats + 20)

    probe = batcher.train_batch(rows[:4])
    M.calibrate_kv_stats(cfg, llm, llm.get_input_embeddings()(probe[1]))

    variants = [
        ("text only (no visual pathway)",
         R.build_system("bridge", clip, llm, cfg, layer_mask=[0, 0, 0, 0])),
        ("caption-style projection",
         R.build_system("caption", clip, llm, cfg)),
        ("cache-bridging, full cache",
         R.build_system("bridge", clip, llm, cfg)),
        ("cache-bridging, 50% top-K",
         R.build_system("bridge", clip, llm, cfg, keep_ratio=0.5)),
        ("cache-bridging, 25% top-K",
         R.build_system("bridge", clip, llm, cfg, keep_ratio=0.25)),
    ]

    report = {"device": common.device_report(), "variants": []}
    for label, system in variants:
        system.eval()
        stats = measure(system, batcher, rows, args.repeats)
        stats["system"] = label
        n_params = sum(p.numel() for p in system.trainable_parameters())
        stats["connector_params"] = n_params
        report["variants"].append(stats)
        print("%-34s %8.1f ms   %5.2f GB   %9d params" % (
            label, stats["latency_ms_mean"], stats["peak_mem_gb"], n_params),
            flush=True)
        del system
        torch.cuda.empty_cache()

    path = common.save_json("inference_cost.json", report)
    print("saved ->", path)


if __name__ == "__main__":
    main()
