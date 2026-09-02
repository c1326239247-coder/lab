# -*- coding: utf-8 -*-
"""Print the paper's main tables from the per-run JSON files."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

ORDER = ["VQAv2", "OK-VQA", "GQA-spatial"]
SYSTEMS = [("blind", "Question only (blind control)"),
           ("caption", "Caption-style projection (baseline)"),
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
    eval_size = None
    for dataset in ORDER:
        for system, _ in SYSTEMS:
            run = load("%s_%s" % (short(dataset), system))
            rows[(dataset, system)] = run
            if run is not None:
                eval_size = run["config"]["eval_size"]

    print("")
    print("Table: VQA accuracy (%%) with 95%% bootstrap CI, %s held-out "
          "questions per dataset" % eval_size)
    print("-" * 92)
    header = "%-36s" % "System"
    for dataset in ORDER:
        header += "%18s" % dataset
    print(header)
    for system, label in SYSTEMS:
        line = "%-36s" % label
        for dataset in ORDER:
            run = rows.get((dataset, system))
            if run is None:
                line += "%18s" % "--"
            else:
                block = run["accuracy_ci95"]
                line += "%18s" % ("%.2f +/-%.2f" % (block["mean"],
                                                    block["half_width"]))
        print(line)

    print("")
    print("Table: control conditions (%), paired bootstrap deltas")
    print("-" * 92)
    print("%-36s%18s%18s%18s" % ("Condition", "VQAv2", "OK-VQA",
                                 "GQA-spatial"))
    for system, label in [("bridge", "cache-bridging"),
                          ("caption", "caption-style")]:
        for key, name in [("accuracy", "correct image"),
                          ("accuracy_shuffled_images", "mismatched image"),
                          ("accuracy_no_visual", "visual pathway off")]:
            line = "%-36s" % ("%s, %s" % (label, name))
            for dataset in ORDER:
                run = rows.get((dataset, system))
                value = None if run is None else run.get(key)
                line += "%18s" % ("--" if value is None else "%.2f" % value)
            print(line)
        for key, name in [("paired_delta_shuffled", "delta vs mismatched"),
                          ("paired_delta_no_visual", "delta vs pathway off")]:
            line = "%-36s" % ("  %s" % name)
            for dataset in ORDER:
                run = rows.get((dataset, system))
                block = None if run is None else run.get(key)
                if not block:
                    line += "%18s" % "--"
                else:
                    line += "%18s" % ("%+.2f%s" % (
                        block["delta"], "" if block["significant"] else " ns"))
            print(line)

    print("")
    print("Table: connector cost on GQA-spatial")
    print("-" * 92)
    print("%-36s%16s%14s%14s" % ("System", "params", "train (s)",
                                 "peak (GB)"))
    for system, label in SYSTEMS:
        run = rows.get(("GQA-spatial", system))
        if run is None:
            continue
        print("%-36s%16d%14.1f%14.2f" % (label, run["connector_params"],
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
            device["gpu"], device["vram_gb"], device["torch"],
            device["cuda"]))


if __name__ == "__main__":
    main()
