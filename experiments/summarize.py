# -*- coding: utf-8 -*-
"""Collect the per-run JSON files into the tables used in the paper."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

ORDER = ["VQAv2", "OK-VQA", "GQA-spatial"]
SYSTEMS = [("caption", "Caption-style projection (baseline)"),
           ("bridge", "Cache-bridging (ours)")]


def load(tag):
    path = os.path.join(common.RESULTS_DIR, "run_%s.json" % tag)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def short(dataset):
    return dataset.replace("-", "").replace("spatial", "")


def main():
    rows = {}
    for dataset in ORDER:
        for system, _ in SYSTEMS:
            tag = "%s_%s" % (short(dataset), system)
            rows[(dataset, system)] = load(tag)

    print("")
    print("Table: VQA accuracy (%), 400 held-out questions per dataset")
    print("-" * 78)
    header = "%-34s" % "System"
    for dataset in ORDER:
        header += "%14s" % dataset
    print(header)
    for system, label in SYSTEMS:
        line = "%-34s" % label
        for dataset in ORDER:
            run = rows.get((dataset, system))
            line += "%14s" % ("--" if run is None else "%.2f" % run["accuracy"])
        print(line)

    print("")
    print("Table: control conditions (%)")
    print("-" * 78)
    print("%-34s%14s%14s%14s" % ("Condition", "VQAv2", "OK-VQA", "GQA-spatial"))
    for key, label in [("accuracy", "correct image"),
                       ("accuracy_shuffled_images", "mismatched image"),
                       ("accuracy_no_visual", "visual cache disabled")]:
        line = "%-34s" % label
        for dataset in ORDER:
            run = rows.get((dataset, "bridge"))
            value = None if run is None else run.get(key)
            line += "%14s" % ("--" if value is None else "%.2f" % value)
        print(line)

    print("")
    print("Table: connector cost")
    print("-" * 78)
    print("%-34s%16s%14s%14s" % ("System", "params", "train (s)", "peak (GB)"))
    for system, label in SYSTEMS:
        run = rows.get(("GQA-spatial", system))
        if run is None:
            continue
        print("%-34s%16d%14.1f%14.2f" % (label, run["connector_params"],
                                         run["train_seconds"],
                                         run["train_peak_mem_gb"]))

    device = None
    for run in rows.values():
        if run is not None:
            device = run["device"]
            break
    if device is not None:
        print("")
        print("Hardware: %s (%s GB), torch %s, CUDA %s" % (
            device["gpu"], device["vram_gb"], device["torch"], device["cuda"]))


if __name__ == "__main__":
    main()
