# -*- coding: utf-8 -*-
"""Build the console summary and every generated figure of the paper."""

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper_figures")

ABLATIONS = [
    ("abl_full", "Full bridge (all four levels)"),
    ("abl_no_geometry", "without 2D geometry bias"),
    ("abl_no_gate", "without adaptive gate"),
    ("abl_gate_fixed", "gate fixed to even blend (matched params)"),
    ("abl_per_layer_scale", "with per-layer K/V scale"),
    ("abl_drop_basic", "without basic level"),
    ("abl_drop_object", "without semantic-object level"),
    ("abl_drop_relation", "without spatial-relation level"),
    ("abl_drop_attribute", "without attribute level"),
    ("abl_only_basic", "basic level only"),
    ("abl_only_object", "semantic-object level only"),
    ("abl_only_relation", "spatial-relation level only"),
    ("abl_only_attribute", "attribute level only"),
    ("abl_keep50", "Top-K compression, keep 50%"),
    ("abl_keep25", "Top-K compression, keep 25%"),
]

CAPACITY = ["cap_rank64", "cap_rank128", "cap_rank256", "cap_rank512",
            "cap_rank1024", "cap_rank2304"]


def load(tag):
    path = os.path.join(common.RESULTS_DIR, "run_%s.json" % tag)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_probe(tag):
    path = os.path.join(common.RESULTS_DIR, "probe_%s.json" % tag)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ablation_table():
    base = load("abl_full")
    print("")
    print("Table: ablations on GQA-spatial")
    print("-" * 84)
    print("%-44s%10s%10s%12s" % ("Variant", "acc (%)", "delta", "params"))
    for tag, label in ABLATIONS:
        run = load(tag)
        if run is None:
            continue
        delta = ("--" if base is None
                 else "%+.2f" % (run["accuracy"] - base["accuracy"]))
        print("%-44s%10.2f%10s%12d" % (label, run["accuracy"], delta,
                                       run["connector_params"]))


def figure_main():
    """Main comparison with bootstrap intervals and the blind control."""
    datasets = [("VQAv2", "VQAv2"), ("OK-VQA", "OKVQA"),
                ("GQA-spatial", "GQA")]
    series = [("blind", "Question only (blind)", "#BBBBBB"),
              ("caption", "Caption-style projection", "#4C72B0"),
              ("bridge", "Cache-bridging", "#DD8452")]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.26
    for offset, (system, label, colour) in zip([-width, 0.0, width], series):
        means, errs = [], []
        for _, short in datasets:
            run = load("%s_%s" % (short, system))
            if run is None:
                means.append(0.0)
                errs.append(0.0)
                continue
            block = run["accuracy_ci95"]
            means.append(block["mean"])
            errs.append(block["half_width"])
        positions = [i + offset for i in range(len(datasets))]
        ax.bar(positions, means, width, label=label, color=colour,
               yerr=errs, capsize=3, ecolor="#444444")
    shuffled = []
    for _, short in datasets:
        run = load("%s_bridge" % short)
        shuffled.append(0.0 if run is None
                        else run["accuracy_shuffled_images"])
    ax.plot(range(len(datasets)), shuffled, "kv", markersize=7,
            label="Cache-bridging, mismatched image")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([name for name, _ in datasets])
    ax.set_title("Connector comparison under an identical training budget\n"
                 "(error bars: 95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, "fig_main_accuracy.png")


def figure_levels():
    labels, values, errs = [], [], []
    for tag, label in [("abl_only_basic", "basic"),
                       ("abl_only_object", "object"),
                       ("abl_only_relation", "relation"),
                       ("abl_only_attribute", "attribute"),
                       ("abl_full", "all four")]:
        run = load(tag)
        if run is None:
            continue
        labels.append(label)
        values.append(run["accuracy"])
        errs.append(run["accuracy_ci95"]["half_width"])
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(labels, values, color="#55A868", yerr=errs, capsize=3,
           ecolor="#444444")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Contribution of each cache level (GQA-spatial)\n"
                 "(error bars: 95% bootstrap CI)")
    _clean(ax)
    return _save(fig, "fig_cache_levels.png")


def figure_capacity():
    points = []
    for tag in CAPACITY:
        run = load(tag)
        if run is not None:
            points.append((run["connector_params"], run["accuracy"],
                           run["accuracy_ci95"]["half_width"]))
    if len(points) < 2:
        return None
    points.sort()
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    es = [p[2] for p in points]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#DD8452", capsize=3,
                label="Cache-bridging")
    baseline = load("GQA_caption")
    blind = load("GQA_blind")
    if baseline is not None:
        ax.axhline(baseline["accuracy"], color="#4C72B0", linestyle="--",
                   label="Caption-style baseline (%s params)"
                         % "{:,}".format(baseline["connector_params"]))
    if blind is not None:
        ax.axhline(blind["accuracy"], color="#999999", linestyle=":",
                   label="Question-only blind control")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable connector parameters (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy against bridge capacity (GQA-spatial)")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, "fig_capacity.png")


def figure_attention():
    probe = load_probe("GQA_bridge")
    if probe is None:
        return None
    mass = probe["attention_mass"]
    per_layer = mass["per_layer_mean"]
    chance = mass["chance_level"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(range(len(per_layer)), per_layer, marker="o", color="#DD8452",
            label="Cache-bridging, injected KV prefix")
    caption = load_probe("GQA_caption")
    if caption is not None:
        other = caption["attention_mass"]["per_layer_mean"]
        ax.plot(range(len(other)), other, marker="s", color="#4C72B0",
                label="Caption-style, soft embedding prefix")
        ax.axhline(caption["attention_mass"]["chance_level"],
                   color="#4C72B0", linestyle=":",
                   label="caption uniform reference")
    ax.axhline(chance, color="#DD8452", linestyle=":",
               label="bridge uniform reference")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Attention mass on visual prefix")
    ax.set_title("How much attention the visual prefix actually receives")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, "fig_attention_mass.png")


def figure_kv_scale():
    probe = load_probe("GQA_bridge")
    if probe is None:
        return None
    profile = probe["kv_profile"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    layers = range(len(profile["host_k_std"]))
    ax.plot(layers, profile["host_k_std"], marker="o", color="#4C72B0",
            label="host keys")
    ax.plot(layers, profile["host_v_std"], marker="s", color="#55A868",
            label="host values")
    if profile.get("bridged"):
        ax.plot([r["layer"] for r in profile["bridged"]],
                [r["k_std"] for r in profile["bridged"]], "--",
                color="#DD8452", label="bridged keys")
        ax.plot([r["layer"] for r in profile["bridged"]],
                [r["v_std"] for r in profile["bridged"]], ":",
                color="#C44E52", label="bridged values")
    ax.set_yscale("log")
    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Standard deviation (log scale)")
    ax.set_title("Key and value scale, host cache against bridged cache")
    ax.legend(frameon=False, fontsize=9)
    _clean(ax)
    return _save(fig, "fig_kv_scale.png")


def main():
    ablation_table()
    for builder in [figure_main, figure_levels, figure_capacity,
                    figure_attention, figure_kv_scale]:
        path = builder()
        print("figure ->", path if path else "(skipped, missing results)")


if __name__ == "__main__":
    main()
