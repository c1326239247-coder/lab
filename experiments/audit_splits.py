# -*- coding: utf-8 -*-
"""Audit image overlap in the question-level train/evaluation split."""

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common


def main():
    common.setup_env()
    report = {}
    for name, loader in common.DATASETS.items():
        rows = loader(5000)
        evaluation = rows[:2000]
        training = rows[2000:]
        eval_images = {hashlib.sha256(row["image"]).digest()
                       for row in evaluation}
        train_images = {hashlib.sha256(row["image"]).digest()
                        for row in training}
        report[name] = {
            "evaluation_questions": len(evaluation),
            "training_questions": len(training),
            "evaluation_unique_images": len(eval_images),
            "training_unique_images": len(train_images),
            "shared_images": len(eval_images & train_images),
        }
        print(name, report[name], flush=True)
    path = common.save_json("split_audit.json", report)
    print("saved ->", path)


if __name__ == "__main__":
    main()
