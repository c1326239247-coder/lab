# -*- coding: utf-8 -*-
"""Rerun every bridge result after prompt-summary consistency fixes."""

import os
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "experiments", "run_vqa.py")
PROBE = os.path.join(ROOT, "experiments", "probe.py")

BASE = ["--train-size", "3000", "--eval-size", "2000", "--steps", "1500",
        "--batch-size", "8", "--lr", "1e-3"]


def run_vqa(dataset, system, tag, extra=()):
    command = [PY, RUN, "--dataset", dataset, "--system", system] + BASE
    command += ["--tag", tag] + list(extra)
    print("===", tag, "===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_script(name):
    print("===", name, "===", flush=True)
    subprocess.run([PY, os.path.join(ROOT, "experiments", name)],
                   cwd=ROOT, check=True)


def main():
    run_vqa("GQA-spatial", "bridge", "GQA_bridge")
    run_vqa("VQAv2", "bridge", "VQAv2_bridge")
    run_vqa("OK-VQA", "bridge", "OKVQA_bridge")
    for seed in (1, 2):
        run_vqa("GQA-spatial", "bridge", "GQA_bridge_seed%d" % seed,
                ["--seed", str(seed)])
    run_script("run_ablations.py")
    run_script("run_capacity.py")
    run_script("run_prefix_sweep.py")
    run_script("run_kv_scale_sweep.py")
    run_vqa("GQA-spatial", "bridge", "Mistral7B_GQA_bridge",
            ["--llm-id", "models/mistral-7b-instruct-v0.2-bnb-4bit"])

    for checkpoint, tag in [("GQA_bridge", "GQA_bridge"),
                            ("kvscale05", "kvscale05"),
                            ("kvscale10", "kvscale10"),
                            ("kvscale20", "kvscale20")]:
        command = [PY, PROBE, "--dataset", "GQA-spatial",
                   "--checkpoint", "checkpoints/%s.pt" % checkpoint,
                   "--tag", tag]
        subprocess.run(command, cwd=ROOT, check=True)
    run_script("run_seeds.py")
    run_script("run_cost.py")
    run_script("compare_systems.py")


if __name__ == "__main__":
    main()
