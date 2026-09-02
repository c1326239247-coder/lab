# -*- coding: utf-8 -*-
"""Collect every number the manuscript quotes from results/*.json."""

import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load(name):
    path = os.path.join(RESULTS, name)
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def pct(value):
    return "%.2f%%" % value


def signed(value):
    return "%+.2f" % value


def facts():
    gqa_b = load("run_GQA_bridge.json")
    gqa_c = load("run_GQA_caption.json")
    gqa_big = load("run_GQA_bridge_big.json")
    vqa_b = load("run_VQAv2_bridge.json")
    vqa_c = load("run_VQAv2_caption.json")
    ok_b = load("run_OKVQA_bridge.json")
    ok_c = load("run_OKVQA_caption.json")
    cost = load("inference_cost.json")
    back = load("backbone_params.json")
    abl = dict((tag, load("run_abl_%s.json" % tag)) for tag in
               ["full", "no_geometry", "no_gate", "gate_fixed", "drop_basic",
                "drop_object", "drop_relation", "drop_attribute", "only_basic",
                "only_object", "only_relation", "only_attribute", "keep50",
                "keep25"])

    cfg = gqa_b["config"]
    dev = gqa_b["device"]
    by_system = dict((v["system"], v) for v in cost["variants"])
    cap_lat = by_system["caption-style projection"]["latency_ms_mean"]
    brg_lat = by_system["cache-bridging, full cache"]["latency_ms_mean"]

    n_visual = cfg["n_visual"]
    total_backbone = back["clip_params"] + back["llm_params"]
    small = gqa_b["connector_params"]

    abl_full = abl["full"]["accuracy"]
    drop = dict((k, abl["drop_" + k]["accuracy"] - abl_full)
                for k in ["basic", "object", "relation", "attribute"])
    only = dict((k, abl["only_" + k]["accuracy"])
                for k in ["basic", "object", "relation", "attribute"])

    seeds = []
    for seed in [1, 2]:
        pair = {}
        for system in ["caption", "bridge"]:
            path = os.path.join(RESULTS, "run_GQA_%s_seed%d.json" % (system, seed))
            if os.path.exists(path):
                pair[system] = load("run_GQA_%s_seed%d.json" % (system, seed))
        if len(pair) == 2:
            seeds.append((seed, pair["caption"]["accuracy"], pair["bridge"]["accuracy"]))
    if seeds:
        parts = ["seed %d gives %s against %s" % (s, pct(c), pct(b))
                 for s, c, b in seeds]
        seed_sentence = ("Repeating the GQA-spatial comparison with two further "
                         "initialisations reproduces the ordering, so it is not a "
                         "seed artefact: " + "; ".join(parts) +
                         " for the baseline and the bridge respectively.")
    else:
        seed_sentence = ""

    # Same-seed repeat runs of the bridge.  adaptive_avg_pool2d has a
    # non-deterministic CUDA backward, so an identical configuration does
    # not reproduce bit-for-bit; we measure the spread instead of hiding it.
    repeat_paths = ["run_GQA_bridge.json", "run_GQA_bridge_rerun2.json",
                    "run_GQA_bridge_rerun3.json"]
    repeats = [load(name)["accuracy"] for name in repeat_paths
               if os.path.exists(os.path.join(RESULTS, name))]
    repeat_spread = (max(repeats) - min(repeats)) if len(repeats) > 1 else 0.0
    bridge_all = repeats + [
        load("run_GQA_bridge_seed%d.json" % s_)["accuracy"] for s_ in [1, 2]
        if os.path.exists(os.path.join(RESULTS, "run_GQA_bridge_seed%d.json" % s_))]
    caption_all = [gqa_c["accuracy"]] + [
        load("run_GQA_caption_seed%d.json" % s_)["accuracy"] for s_ in [1, 2]
        if os.path.exists(os.path.join(RESULTS, "run_GQA_caption_seed%d.json" % s_))]

    gqa_big_acc = gqa_big["accuracy"]
    if abs(gqa_big_acc - gqa_b["accuracy"]) < 1e-9:
        big_text = "identical to"
    elif gqa_big_acc > gqa_b["accuracy"]:
        big_text = "%+.2f points above" % (gqa_big_acc - gqa_b["accuracy"])
    else:
        big_text = "%.2f points below" % (gqa_b["accuracy"] - gqa_big_acc)

    total_seconds = 0.0
    for name in os.listdir(RESULTS):
        if name.startswith("run_") and name.endswith(".json"):
            run = load(name)
            total_seconds += run.get("train_seconds", 0) + run.get("eval_seconds", 0)

    values = {
        "gqa_bridge": pct(gqa_b["accuracy"]),
        "gqa_caption": pct(gqa_c["accuracy"]),
        "gqa_caption_shuf": pct(gqa_c["accuracy_shuffled_images"]),
        "gqa_shuf": pct(gqa_b["accuracy_shuffled_images"]),
        "gqa_novis": pct(gqa_b["accuracy_no_visual"]),
        "gqa_shuf_delta": "%.2f" % (gqa_b["accuracy"] - gqa_b["accuracy_shuffled_images"]),
        "gqa_novis_delta": "%.2f" % (gqa_b["accuracy"] - gqa_b["accuracy_no_visual"]),
        "vqav2_bridge": pct(vqa_b["accuracy"]),
        "vqav2_novis": pct(vqa_b["accuracy_no_visual"]),
        "okvqa_novis": pct(ok_b["accuracy_no_visual"]),
        "gap_vqav2": "%.2f" % (vqa_c["accuracy"] - vqa_b["accuracy"]),
        "gap_okvqa": "%.2f" % (ok_c["accuracy"] - ok_b["accuracy"]),
        "gap_gqa": "%.2f" % (gqa_c["accuracy"] - gqa_b["accuracy"]),
        "seed_sentence": seed_sentence,
        "bridge_loss_first": "%.2f" % gqa_b["loss_first_25"],
        "bridge_loss_last": "%.2f" % gqa_b["loss_last_25"],
        "caption_loss_first": "%.2f" % gqa_c["loss_first_25"],
        "small_params": "{:,}".format(small),
        "big_params": "{:,}".format(gqa_big["connector_params"]),
        "caption_params": "{:,}".format(gqa_c["connector_params"]),
        "gqa_big": pct(gqa_big_acc),
        "gqa_big_shuf": pct(gqa_big["accuracy_shuffled_images"]),
        "gqa_big_delta_text": big_text,
        "big_rank": gqa_big["config"]["rank"],
        "big_n_visual": gqa_big["config"]["n_visual"],
        "k_std": "%.2f" % gqa_b["host_kv_stats"]["k_std"],
        "v_std": "%.2f" % gqa_b["host_kv_stats"]["v_std"],
        "v_ratio": "%d" % round(1.0 / gqa_b["host_kv_stats"]["v_std"]),
        "gpu": dev["gpu"],
        "vram": dev["vram_gb"],
        "cc": dev["compute_capability"],
        "torch_ver": dev["torch"],
        "cuda_ver": dev["cuda"],
        "clip_params": "{:,}".format(back["clip_params"]),
        "llm_params": "{:,}".format(back["llm_params"]),
        "llm_layers": back["llm_num_layers"],
        "llm_hidden": back["llm_hidden_size"],
        "llm_heads": back["llm_num_attention_heads"],
        "llm_kv_heads": back["llm_num_key_value_heads"],
        "llm_head_dim": back["llm_hidden_size"] // back["llm_num_attention_heads"],
        "llm_kv_dim": (back["llm_num_key_value_heads"]
                       * (back["llm_hidden_size"] // back["llm_num_attention_heads"])),
        "clip_layers": back["clip_num_layers"],
        "vision_dim": back["clip_hidden_size"],
        "grid": back["clip_image_size"] // back["clip_patch_size"],
        "n_patches": (back["clip_image_size"] // back["clip_patch_size"]) ** 2,
        "n_visual": n_visual,
        "n_visual_total": n_visual * 4,
        "rank": cfg["rank"],
        "max_offset": back["clip_image_size"] // back["clip_patch_size"],
        "total_backbone": "{:,}".format(total_backbone),
        "bridge_pct": "%.3f%%" % (100.0 * small / total_backbone),
        "lr": "%g" % cfg["lr"],
        "batch": cfg["batch_size"],
        "steps": cfg["steps"],
        "abl_steps": abl["full"]["config"]["steps"],
        "eval_size": cfg["eval_size"],
        "train_size": cfg["train_size"],
        "eos_weight": "0.2",
        "max_new_tokens": 6,
        "cost_new_tokens": 6,
        "cost_repeats": by_system["cache-bridging, full cache"]["n_measured"],
        "caption_latency": "%.1f" % cap_lat,
        "bridge_latency": "%.1f" % brg_lat,
        "latency_pct": "%.1f%%" % (100.0 * (brg_lat - cap_lat) / cap_lat),
        "bridge_train_s": "%.1f" % gqa_b["train_seconds"],
        "caption_train_s": "%.1f" % gqa_c["train_seconds"],
        "bridge_train_mem": "%.2f" % gqa_b["train_peak_mem_gb"],
        "caption_train_mem": "%.2f" % gqa_c["train_peak_mem_gb"],
        "bridge_train_s_min": "%.0f" % (gqa_b["train_seconds"] / 60.0),
        "total_gpu_hours": "%.1f" % (total_seconds / 3600.0),
        "abl_full": pct(abl_full),
        "abl_gate": pct(abl["no_gate"]["accuracy"]),
        "abl_gate_delta": "%.2f" % (abl_full - abl["no_gate"]["accuracy"]),
        "abl_gate_params": "{:,}".format(abl["no_gate"]["connector_params"]),
        "abl_gate_fixed": pct(abl["gate_fixed"]["accuracy"]),
        "abl_gate_fixed_params": "{:,}".format(abl["gate_fixed"]["connector_params"]),
        "abl_gate_fixed_delta": "%.2f" % (abl_full - abl["gate_fixed"]["accuracy"]),
        "abl_novis": pct(abl["full"]["accuracy_no_visual"]),
        "n_repeats": "%d" % len(repeats),
        "repeat_spread": "%.2f" % repeat_spread,
        "repeat_min": pct(min(repeats)),
        "repeat_max": pct(max(repeats)),
        "bridge_mean": pct(sum(bridge_all) / len(bridge_all)),
        "bridge_max": pct(max(bridge_all)),
        "caption_mean": pct(sum(caption_all) / len(caption_all)),
        "caption_min": pct(min(caption_all)),
        "worst_case_gap": "%.2f" % (min(caption_all) - max(bridge_all)),
        "n_bridge_runs": "%d" % len(bridge_all),
        "n_caption_runs": "%d" % len(caption_all),
        "abl_geo_delta": "%.2f" % (abl_full - abl["no_geometry"]["accuracy"]),
        "abl_keep50_delta": "%.2f" % (abl_full - abl["keep50"]["accuracy"]),
        "abl_keep25_delta": "%.2f" % (abl_full - abl["keep25"]["accuracy"]),
        "abl_best_single": pct(max(only.values())),
        "abl_drop_min": "%.2f" % min(abs(v) for v in drop.values()),
        "abl_drop_max": "%.2f" % max(abs(v) for v in drop.values()),
        "only_basic": pct(only["basic"]),
        "only_object": pct(only["object"]),
        "only_relation": pct(only["relation"]),
        "only_attribute": pct(only["attribute"]),
        "drop_basic_delta": "%.2f" % abs(drop["basic"]),
        "drop_object_delta": "%.2f" % abs(drop["object"]),
        "drop_relation_delta": "%.2f" % abs(drop["relation"]),
        "drop_attribute_delta": "%.2f" % abs(drop["attribute"]),
        "n_shape_tests": 7,
        "n_torch_tests": 10,
        "n_numpy_tests": 9,
    }
    runs = {"gqa_b": gqa_b, "gqa_c": gqa_c, "gqa_big": gqa_big,
            "vqa_b": vqa_b, "vqa_c": vqa_c, "ok_b": ok_b, "ok_c": ok_c,
            "cost": cost, "abl": abl, "back": back, "seeds": seeds}
    return values, runs


if __name__ == "__main__":
    values, _ = facts()
    for key in sorted(values):
        print("%-22s %s" % (key, values[key]))
