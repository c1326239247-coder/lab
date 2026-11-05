# -*- coding: utf-8 -*-
"""Capacity sweep: is the gap a matter of connector size?

The caption baseline trains 6.30M parameters and the default bridge 0.82M,
so the headline comparison is not parameter-matched.  This sweep grows the
bridge over roughly an order of magnitude, up to and past the baseline's
budget, under the main-comparison protocol.
"""

import os
import subprocess
import sys

PY = sys.executable
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--dataset", "GQA-spatial", "--system", "bridge",
        "--train-size", "3000", "--eval-size", "2000", "--steps", "1500",
        "--batch-size", "8", "--lr", "1e-3"]

# (tag, rank, visual tokens).  The last point overshoots the baseline's
# 6.30M so the curve brackets it instead of stopping short.
POINTS = [
    ("cap_rank64", "64", "36"),
    ("cap_rank128", "128", "36"),
    ("cap_rank256", "256", "36"),
    ("cap_rank512", "512", "64"),
    ("cap_rank1024", "1024", "64"),
    ("cap_rank2304", "2304", "64"),
]


def main():
    for tag, rank, n_visual in POINTS:
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, SCRIPT] + BASE
                       + ["--rank", rank, "--n-visual", n_visual,
                          "--tag", tag], check=True)
    print("ALL CAPACITY RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
