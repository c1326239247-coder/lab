# -*- coding: utf-8 -*-
"""Run the main comparison: three datasets x three systems.

The evaluation split is 2000 questions.  With 400 questions the 95%
bootstrap interval on a single accuracy is about +/-4 points, which is
wider than most of the differences discussed in the paper; 2000 questions
brings it under +/-2 and is affordable because only the connector trains.
"""

import os
import subprocess
import sys

RUNS = []
for dataset, prefix in [("GQA-spatial", "GQA"), ("VQAv2", "VQAv2"),
                        ("OK-VQA", "OKVQA")]:
    for system in ["blind", "caption", "bridge"]:
        RUNS.append((dataset, system, "%s_%s" % (prefix, system)))

PY = sys.executable
SCRIPT = os.path.join("experiments", "run_vqa.py")

BASE = ["--train-size", "3000", "--eval-size", "2000", "--steps", "1500",
        "--batch-size", "8", "--lr", "1e-3"]


def main():
    for dataset, system, tag in RUNS:
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, SCRIPT] + BASE
                       + ["--dataset", dataset, "--system", system,
                          "--tag", tag], check=True)
    print("ALL MAIN RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
