# -*- coding: utf-8 -*-
"""Run the main comparison: three datasets x two connector designs."""

import os
import subprocess
import sys

RUNS = [
    ("GQA-spatial", "caption", "GQA_caption"),
    ("GQA-spatial", "bridge", "GQA_bridge"),
    ("VQAv2", "caption", "VQAv2_caption"),
    ("VQAv2", "bridge", "VQAv2_bridge"),
    ("OK-VQA", "caption", "OKVQA_caption"),
    ("OK-VQA", "bridge", "OKVQA_bridge"),
]

PY = os.path.join(".venv", "Scripts", "python.exe")
SCRIPT = os.path.join("experiments", "run_vqa.py")


def main():
    for dataset, system, tag in RUNS:
        print("=== %s ===" % tag, flush=True)
        cmd = [PY, SCRIPT,
               "--dataset", dataset,
               "--system", system,
               "--train-size", "3000",
               "--eval-size", "400",
               "--steps", "1500",
               "--batch-size", "8",
               "--lr", "1e-3",
               "--tag", tag]
        subprocess.run(cmd, check=True)
    print("ALL MAIN RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
