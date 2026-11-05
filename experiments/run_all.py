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

PY = sys.executable

STAGES = [
    ("main comparison", [PY, os.path.join("experiments", "run_main.py")]),
    ("seed repeats", [PY, os.path.join("experiments", "run_seeds.py")]),
    ("ablations", [PY, os.path.join("experiments", "run_ablations.py")]),
    ("capacity sweep", [PY, os.path.join("experiments", "run_capacity.py")]),
    ("matched prefix sweep", [PY, os.path.join("experiments",
                                               "run_prefix_sweep.py")]),
    ("attention-allocation sweep", [PY, os.path.join(
        "experiments", "run_kv_scale_sweep.py")]),
    ("inference cost", [PY, os.path.join("experiments", "run_cost.py")]),
    ("Mistral-7B sensitivity check", [PY, os.path.join(
        "experiments", "run_mistral.py")]),
    ("paired system comparisons", [PY, os.path.join(
        "experiments", "compare_systems.py")]),
    ("question-level split audit", [PY, os.path.join(
        "experiments", "audit_splits.py")]),
    ("backbone parameters", [PY, os.path.join("experiments",
                                              "backbone_params.py")]),
    ("bridge attention probe", [PY, os.path.join("experiments", "probe.py"),
                               "--checkpoint", "checkpoints/GQA_bridge.pt"]),
    ("caption attention probe", [PY, os.path.join("experiments", "probe.py"),
                                "--checkpoint", "checkpoints/GQA_caption.pt"]),
    ("console summary", [PY, os.path.join("experiments", "summarize.py")]),
    ("paper figures", [PY, os.path.join("experiments", "report.py")]),
    ("Figure 1 labels", [PY, os.path.join("rewrite",
                                         "annotate_figure1.py")]),
    ("README", [PY, os.path.join("rewrite", "build_readme.py")]),
    ("revised manuscript", [PY, os.path.join("rewrite", "build_docx.py")]),
    ("reviewer response", [PY, os.path.join("rewrite", "build_response.py")]),
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
