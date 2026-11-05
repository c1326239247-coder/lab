# -*- coding: utf-8 -*-
"""Matched-prefix-length comparison on GQA-spatial.

The bridge injects four tokens per level, so `n_visual` 6, 9, 18 and 36
give prefixes of 24, 36, 72 and 144 tokens.  The caption-style baseline
uses exactly the same total number of prefix tokens.
"""

import os
import subprocess
import sys

PY = sys.executable
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--dataset", "GQA-spatial", "--train-size", "3000",
        "--eval-size", "2000", "--steps", "800", "--batch-size", "8",
        "--lr", "1e-3"]


def main():
    for n_visual in (6, 9, 18, 36):
        for system in ("bridge", "caption"):
            tag = "prefix%s_%s" % (n_visual, system)
            if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
                print("=== %s (already present, skipped) ===" % tag, flush=True)
                continue
            print("=== %s ===" % tag, flush=True)
            subprocess.run([PY, SCRIPT] + BASE
                           + ["--system", system, "--n-visual", str(n_visual),
                              "--tag", tag], check=True)
    print("PREFIX SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
