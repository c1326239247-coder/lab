# -*- coding: utf-8 -*-
"""Build the ablation table and the figures used in the paper."""

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


def load(tag):
    path = os.path.join(common.RESULTS_DIR, "run_%s.json" % tag)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def ablation_table():
    base = load("abl_full")
    print("")
    print("Table: ablations on GQA-spatial (800 steps, 400 held-out questions)")
    print("-" * 76)
    print("%-38s%10s%10s%10s" % ("Variant", "acc (%)", "delta", "params"))
    rows = []
    for tag, label in ABLATIONS:
        run = load(tag)
        if run is None:
            continue
        delta = ("--" if base is None
                 else "%+.2f" % (run["accuracy"] - base["accuracy"]))
        print("%-38s%10.2f%10s%10d" % (label, run["accuracy"], delta,
                                       run["connector_params"]))
        rows.append((label, run["accuracy"], run["connector_params"]))
    return rows


def figure_main():
    datasets = ["VQAv2", "OK-VQA", "GQA-spatial"]
    caption, bridge, control = [], [], []
    for dataset in datasets:
        short = dataset.replace("-", "").replace("spatial", "")
        cap = load("%s_caption" % short)
        brd = load("%s_bridge" % short)
        caption.append(0 if cap is None else cap["accuracy"])
        bridge.append(0 if brd is None else brd["accuracy"])
        control.append(0 if brd is None else brd["accuracy_shuffled_images"])

    x = range(len(datasets))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([i - width for i in x], caption, width,
           label="Caption-style projection", color="#4C72B0")
    ax.bar(list(x), bridge, width, label="Cache-bridging", color="#DD8452")
    ax.bar([i + width for i in x], control, width,
           label="Cache-bridging, mismatched image", color="#CCCCCC")
    ax.set_ylabel("VQA accuracy (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_title("Connector comparison under an identical training budget")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "fig_main_accuracy.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_levels():
    labels, values = [], []
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
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, values, color="#55A868")
    ax.set_ylabel("VQA accuracy (%)")
    ax.set_title("Contribution of each cache level (GQA-spatial)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "fig_cache_levels.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    ablation_table()
    for path in [figure_main(), figure_levels()]:
        if path:
            print("figure ->", path)


if __name__ == "__main__":
    main()
