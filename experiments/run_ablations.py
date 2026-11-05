# -*- coding: utf-8 -*-
"""Ablations on the GQA spatial subset.

Each variant changes exactly one component of the bridge and is trained
with an identical budget, so the accuracy delta is attributable to that
component.  The evaluation split is 2000 questions, the same as the main
comparison, so ablation deltas can be read against a bootstrap interval
of roughly +/-2 points rather than +/-4.
"""

import os
import subprocess
import sys

PY = sys.executable
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--dataset", "GQA-spatial", "--system", "bridge",
        "--train-size", "3000", "--eval-size", "2000",
        "--steps", "800", "--batch-size", "8", "--lr", "1e-3"]

VARIANTS = [
    ("abl_full", []),
    ("abl_no_geometry", ["--no-geometry"]),
    ("abl_no_gate", ["--no-gate"]),
    ("abl_gate_fixed", ["--gate-mode", "fixed"]),
    ("abl_per_layer_scale", ["--per-layer-scale"]),
    ("abl_only_basic", ["--layer-mask", "1,0,0,0"]),
    ("abl_only_object", ["--layer-mask", "0,1,0,0"]),
    ("abl_only_relation", ["--layer-mask", "0,0,1,0"]),
    ("abl_only_attribute", ["--layer-mask", "0,0,0,1"]),
    ("abl_drop_basic", ["--layer-mask", "0,1,1,1"]),
    ("abl_drop_object", ["--layer-mask", "1,0,1,1"]),
    ("abl_drop_relation", ["--layer-mask", "1,1,0,1"]),
    ("abl_drop_attribute", ["--layer-mask", "1,1,1,0"]),
    ("abl_keep50", ["--keep-ratio", "0.5"]),
    ("abl_keep25", ["--keep-ratio", "0.25"]),
]


def main():
    for tag, extra in VARIANTS:
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, SCRIPT] + BASE + extra + ["--tag", tag], check=True)
    print("ALL ABLATIONS DONE", flush=True)


if __name__ == "__main__":
    main()
