# -*- coding: utf-8 -*-
"""Seed repeats: how much of any gap survives re-initialisation?

Both systems are repeated over the same three seeds so their run-to-run
spread is measured under identical conditions.  Seed 0 is the main run.
"""

import os
import subprocess

PY = os.path.join(".venv", "Scripts", "python.exe")
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--dataset", "GQA-spatial", "--train-size", "3000",
        "--eval-size", "2000", "--steps", "1500", "--batch-size", "8",
        "--lr", "1e-3"]

RUNS = []
for seed in [1, 2]:
    for system in ["caption", "bridge"]:
        RUNS.append((system, seed, "GQA_%s_seed%d" % (system, seed)))


def main():
    for system, seed, tag in RUNS:
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, SCRIPT] + BASE
                       + ["--system", system, "--seed", str(seed),
                          "--tag", tag], check=True)
    print("ALL SEED RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
