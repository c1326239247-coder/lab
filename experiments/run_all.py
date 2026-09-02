# -*- coding: utf-8 -*-
"""Run every experiment the paper reports, in order, on one GPU.

Existing result files are skipped, so the queue can be interrupted and
restarted.  Order is chosen so the numbers the manuscript needs first
(main comparison) land first.
"""

import os
import subprocess
import sys
import time

PY = os.path.join(".venv", "Scripts", "python.exe")

STAGES = [
    ("main comparison", [PY, os.path.join("experiments", "run_main.py")]),
    ("seed repeats", [PY, os.path.join("experiments", "run_seeds.py")]),
    ("ablations", [PY, os.path.join("experiments", "run_ablations.py")]),
    ("capacity sweep", [PY, os.path.join("experiments", "run_capacity.py")]),
    ("inference cost", [PY, os.path.join("experiments", "run_cost.py")]),
    ("backbone parameters", [PY, os.path.join("experiments",
                                              "backbone_params.py")]),
]


def main():
    for name, cmd in STAGES:
        print("########## %s ##########" % name, flush=True)
        start = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("STAGE FAILED: %s" % name, flush=True)
            sys.exit(result.returncode)
        print("%s finished in %.1f min" % (name, (time.time() - start) / 60.0),
              flush=True)
    print("ALL STAGES DONE", flush=True)


if __name__ == "__main__":
    main()
