# -*- coding: utf-8 -*-
"""Attention-allocation sweep: scale bridged K and V simultaneously."""

import os
import subprocess
import sys

PY = sys.executable
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--dataset", "GQA-spatial", "--system", "bridge",
        "--train-size", "3000", "--eval-size", "2000", "--steps", "800",
        "--batch-size", "8", "--lr", "1e-3", "--n-visual", "36"]


def main():
    for scale in (0.5, 1.0, 2.0):
        tag = "kvscale%s" % str(scale).replace(".", "")
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, SCRIPT] + BASE
                       + ["--bridge-kv-scale", str(scale), "--tag", tag],
                       check=True)
    print("KV-SCALE SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
