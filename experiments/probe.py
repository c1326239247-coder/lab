# -*- coding: utf-8 -*-
"""Diagnostics on a trained connector: attention mass and KV statistics.

Two measurements are produced, both on the evaluation split and both with
the connector weights that were actually scored:

  attention mass  for every decoder layer and head, the fraction of the
                  softmax probability that text tokens assign to the
                  injected visual prefix.  If the injected cache were
                  being read, this fraction should be well above the
                  share the prefix would get by chance.

  KV statistics   per-layer standard deviation of the host model's own
                  keys and values next to the bridged ones, which shows
                  whether the injected tensors sit on the host's scale at
                  every depth or only on average.
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import model as M
import run_vqa as R

common.setup_env()


@torch.no_grad()
def attention_mass(system, llm, batcher, rows, batch_size=4, max_batches=25):
    """Fraction of text-token attention that lands on the visual prefix."""
    layers = llm.config.num_hidden_layers
    heads = llm.config.num_attention_heads
    total = torch.zeros(layers, heads, dtype=torch.float64)
    chance = []
    seen = 0
    for start in range(0, min(len(rows), batch_size * max_batches), batch_size):
        chunk = rows[start:start + batch_size]
        if not chunk:
            break
        px, ids, mask = batcher.eval_batch(chunk)
        embed = llm.get_input_embeddings()(ids)
        if system.mode == "bridge":
            summary = embed.mean(dim=1, keepdim=True)
            k, v = system.visual_kv(px, summary)
            n_prefix = k.shape[2]
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
            for idx in range(layers):
                lk, lv = M.scale_for_layer(k, v, system.layer_scales, idx)
                cache.update(lk, lv, idx)
            visual_mask = torch.ones(embed.shape[0], n_prefix,
                                     dtype=mask.dtype, device=mask.device)
            full_mask = torch.cat([visual_mask, mask], dim=1)
            positions = (full_mask.cumsum(dim=1) - 1).clamp(min=0)[:, n_prefix:]
            out = llm(inputs_embeds=embed, attention_mask=full_mask,
                      past_key_values=cache, position_ids=positions,
                      use_cache=False, output_attentions=True)
            text_mask = mask
        else:
            prefix = system.visual_prefix(px)
            n_prefix = prefix.shape[1]
            inputs = torch.cat([prefix, embed], dim=1)
            pad = torch.ones(prefix.shape[:2], dtype=mask.dtype,
                             device=mask.device)
            full_mask = torch.cat([pad, mask], dim=1)
            out = llm(inputs_embeds=inputs, attention_mask=full_mask,
                      output_attentions=True)
            text_mask = mask
        attns = out.attentions
        if attns is None or attns[0] is None:
            raise RuntimeError("the model did not return attention maps")
        for idx, attn in enumerate(attns):
            # attn: (B, H, Q, K).  For the bridge the query axis covers only
            # the text tokens; for the baseline it covers prefix+text, so the
            # prefix queries are dropped to keep the two comparable.
            if attn.shape[2] != text_mask.shape[1]:
                attn = attn[:, :, n_prefix:, :]
            share = attn[:, :, :, :n_prefix].sum(dim=-1)
            weight = text_mask.unsqueeze(1).to(share.dtype)
            num = (share * weight).sum(dim=(0, 2)).double().cpu()
            total[idx] += num
        seen += float(text_mask.sum())
        # Chance level: a uniform reader would spread its attention over the
        # prefix plus the text tokens visible so far.
        lengths = text_mask.sum(dim=1)
        for length in lengths.tolist():
            positions_seen = torch.arange(1, int(length) + 1, dtype=torch.float64)
            chance.append(float((n_prefix / (n_prefix + positions_seen)).mean()))
    per_layer_head = (total / max(seen, 1.0)).tolist()
    per_layer = [sum(row) / len(row) for row in per_layer_head]
    return {
        "n_prefix": int(n_prefix),
        "text_tokens_scored": int(seen),
        "chance_level": round(sum(chance) / len(chance), 4),
        "per_layer_mean": [round(x, 5) for x in per_layer],
        "per_layer_head": [[round(x, 5) for x in row] for row in per_layer_head],
        "overall_mean": round(sum(per_layer) / len(per_layer), 5),
    }


@torch.no_grad()
def kv_profile(system, llm, batcher, rows, batch_size=4, max_batches=8):
    """Per-layer std of the host KV cache and of the bridged cache."""
    from transformers.cache_utils import DynamicCache

    host_k, host_v = [], []
    for start in range(0, min(len(rows), batch_size * max_batches), batch_size):
        chunk = rows[start:start + batch_size]
        if not chunk:
            break
        _, ids, mask, _ = batcher.train_batch(chunk)
        embed = llm.get_input_embeddings()(ids)
        cache = DynamicCache()
        llm(inputs_embeds=embed, attention_mask=mask,
            past_key_values=cache, use_cache=True)
        host_k.append([layer.keys.float().std().item() for layer in cache.layers])
        host_v.append([layer.values.float().std().item() for layer in cache.layers])
    n = len(host_k)
    host_k = [sum(col) / n for col in zip(*host_k)]
    host_v = [sum(col) / n for col in zip(*host_v)]

    bridged = None
    if system is not None and system.mode == "bridge":
        px, ids, mask = batcher.eval_batch(rows[:batch_size])
        embed = llm.get_input_embeddings()(ids)
        k, v = system.visual_kv(px, embed.mean(dim=1, keepdim=True))
        rows_out = []
        for idx in range(llm.config.num_hidden_layers):
            lk, lv = M.scale_for_layer(k, v, system.layer_scales, idx)
            rows_out.append({"layer": idx,
                             "k_std": round(lk.float().std().item(), 4),
                             "v_std": round(lv.float().std().item(), 4)})
        bridged = rows_out
    return {"host_k_std": [round(x, 4) for x in host_k],
            "host_v_std": [round(x, 4) for x in host_v],
            "bridged": bridged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GQA-spatial")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint written by run_vqa.py, e.g. "
                         "checkpoints/GQA_bridge.pt")
    ap.add_argument("--eval-size", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    common.set_seed(0)
    blob = torch.load(os.path.join(common.REPO_ROOT, args.checkpoint),
                      map_location="cpu", weights_only=False)
    cfg_args = blob["config"]
    kind = cfg_args["system"]
    print("probing %s (%s)" % (args.checkpoint, kind), flush=True)

    rows = common.DATASETS[args.dataset](
        cfg_args["train_size"] + cfg_args["eval_size"])
    eval_rows = rows[:args.eval_size]

    clip, llm, tok, proc = R.load_backbones()
    cfg = M.BridgeCfg(clip, llm, rank=cfg_args["rank"],
                      n_visual=cfg_args["n_visual"])
    batcher = R.Batcher(tok, proc)

    kwargs = {}
    if kind not in ("caption", "blind"):
        probe = batcher.train_batch(rows[cfg_args["eval_size"]:][:4])
        M.calibrate_kv_stats(cfg, llm, llm.get_input_embeddings()(probe[1]))
        mask = None
        if cfg_args.get("layer_mask"):
            mask = [float(x) for x in cfg_args["layer_mask"].split(",")]
        kwargs = {"inject_mode": cfg_args["inject_mode"],
                  "use_geometry": not cfg_args["no_geometry"],
                  "use_gate": not cfg_args["no_gate"],
                  "gate_mode": cfg_args["gate_mode"],
                  "layer_mask": mask,
                  "keep_ratio": cfg_args["keep_ratio"],
                  "per_layer_scale": cfg_args.get("per_layer_scale", False)}
    system = R.build_system(kind, clip, llm, cfg, **kwargs)
    missing, unexpected = system.load_state_dict(blob["state"], strict=False)
    trained = [k for k in blob["state"]]
    print("loaded %d trained tensors (%d unexpected)" % (len(trained),
                                                         len(unexpected)),
          flush=True)
    system.eval()

    # Attention maps are only materialised by the eager kernel; the fused
    # SDPA path never forms the full matrix.
    llm.set_attn_implementation("eager")

    mass = attention_mass(system, llm, batcher, eval_rows)
    print("attention on visual prefix: %.4f (chance %.4f)" % (
        mass["overall_mean"], mass["chance_level"]), flush=True)
    profile = kv_profile(system, llm, batcher, eval_rows)

    tag = args.tag or os.path.splitext(os.path.basename(args.checkpoint))[0]
    payload = {"checkpoint": args.checkpoint, "system": kind,
               "dataset": args.dataset, "device": common.device_report(),
               "attention_mass": mass, "kv_profile": profile}
    path = common.save_json("probe_%s.json" % tag, payload)
    print("saved ->", path, flush=True)


if __name__ == "__main__":
    main()
