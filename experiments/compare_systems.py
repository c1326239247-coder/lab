# -*- coding: utf-8 -*-
"""Paired bootstrap comparisons between systems on the same questions."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common


def load(tag):
    path = os.path.join(common.RESULTS_DIR, "run_%s.json" % tag)
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def compare(output, name, tags):
    """Store left-minus-right accuracy with a paired 95% interval."""
    left_tag, right_tag = tags
    left = load(left_tag)
    right = load(right_tag)
    block = common.paired_delta_ci(left["per_question_scores"],
                                   right["per_question_scores"])
    if block is None:
        raise RuntimeError("unpaired or missing scores for %s" % name)
    block["left"] = left_tag
    block["right"] = right_tag
    output[name] = block


def main():
    output = {}
    for dataset in ("VQAv2", "OKVQA", "GQA"):
        compare(output, "%s_caption_minus_bridge" % dataset,
                ("%s_caption" % dataset, "%s_bridge" % dataset))
        compare(output, "%s_bridge_minus_blind" % dataset,
                ("%s_bridge" % dataset, "%s_blind" % dataset))

    capacity_tags = ["cap_rank64", "cap_rank128", "cap_rank256",
                     "cap_rank512", "cap_rank1024", "cap_rank2304"]
    capacity = [(tag, load(tag)) for tag in capacity_tags]
    best_tag, _ = max(capacity, key=lambda item: item[1]["accuracy"])
    compare(output, "capacity_best_minus_caption", (best_tag, "GQA_caption"))

    for tag in ("6", "9", "18", "36"):
        compare(output, "prefix%s_caption_minus_bridge" % tag,
                ("prefix%s_caption" % tag, "prefix%s_bridge" % tag))
    for tag in ("05", "20"):
        compare(output, "kvscale%s_minus_kvscale10" % tag,
                ("kvscale%s" % tag, "kvscale10"))
    compare(output, "Mistral7B_caption_minus_bridge",
            ("Mistral7B_GQA_caption", "Mistral7B_GQA_bridge"))

    path = common.save_json("paired_comparisons.json",
                            {"n_boot": 10000, "comparisons": output})
    print("saved ->", path)


if __name__ == "__main__":
    main()
