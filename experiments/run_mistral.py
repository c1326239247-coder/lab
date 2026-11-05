# -*- coding: utf-8 -*-
"""Run the larger-decoder sensitivity check on GQA-spatial."""

import os
import subprocess
import sys

PY = sys.executable
LLM = "models/mistral-7b-instruct-v0.2-bnb-4bit"
BASE = ["--dataset", "GQA-spatial", "--train-size", "3000",
        "--eval-size", "2000", "--steps", "1500", "--batch-size", "8",
        "--lr", "1e-3", "--llm-id", LLM]


def main():
    for system in ("blind", "caption", "bridge"):
        tag = "Mistral7B_GQA_%s" % system
        if os.path.exists(os.path.join("results", "run_%s.json" % tag)):
            print("=== %s (already present, skipped) ===" % tag, flush=True)
            continue
        print("=== %s ===" % tag, flush=True)
        subprocess.run([PY, os.path.join("experiments", "run_vqa.py")]
                       + BASE + ["--system", system, "--tag", tag],
                       check=True)
    print("MISTRAL-7B SENSITIVITY CHECK DONE", flush=True)


if __name__ == "__main__":
    main()
