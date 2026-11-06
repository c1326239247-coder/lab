# -*- coding: utf-8 -*-
"""Data tables of the manuscript, read straight from results/*.json.

Split out of build_docx.py: this module decides what each table contains,
build_docx.py decides where it goes in the document."""

import io
import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


def load(name):
    """Read one raw result file, without relying on a shadowed stdlib name."""
    with io.open(os.path.join(RESULTS, name), encoding="utf-8") as handle:
        return json.load(handle)


def build_tables(values, runs):
    """The data tables, all read straight from results/*.json."""
    abl = runs["abl"]
    back = runs["back"]
    n = back["clip_num_layers"]
    taps = [("basic", "n/6", max(1, n // 6),
             "early texture and edge statistics"),
            ("semantic_object", "n/2", n // 2, "mid-level object parts"),
            ("spatial_relation", "3n/4", int(n * 0.75),
             "layout and relational features"),
            ("attribute", "n-2", n - 2,
             "late, globally contextualised attributes")]

    levels = [["Cache level", "Tap depth", "CLIP layer",
               "Hypothesised content"]]
    for name, formula, idx, meaning in taps:
        levels.append([name, formula, idx, meaning])

    def acc_ci(run):
        block = run["accuracy_ci95"]
        return "%.2f (%.2f-%.2f)" % (block["mean"], block["low"],
                                     block["high"])

    main = [["Connector", "Trainable params", "VQAv2", "OK-VQA",
             "GQA-spatial"]]
    main.append(["Question only (blind control)",
                 "{:,}".format(runs["gqa_n"]["connector_params"]),
                 acc_ci(runs["vqa_n"]), acc_ci(runs["ok_n"]),
                 acc_ci(runs["gqa_n"])])
    main.append(["Caption-style projection (baseline)",
                 "{:,}".format(runs["vqa_c"]["connector_params"]),
                 acc_ci(runs["vqa_c"]), acc_ci(runs["ok_c"]),
                 acc_ci(runs["gqa_c"])])
    main.append(["Cache-bridging (this work)",
                 "{:,}".format(runs["vqa_b"]["connector_params"]),
                 acc_ci(runs["vqa_b"]), acc_ci(runs["ok_b"]),
                 acc_ci(runs["gqa_b"])])

    def cell(run, key):
        value = run.get(key)
        return "\u2014" if value is None else "%.2f" % value

    def paired(run, key):
        block = run.get(key)
        if not block:
            return "\u2014"
        return "%+.2f (%+.2f to %+.2f)%s" % (
            block["delta"], block["low"], block["high"],
            "" if block["significant"] else " n.s.")

    control = [["Condition", "VQAv2", "OK-VQA", "GQA-spatial"]]
    keys = ["vqa_b", "ok_b", "gqa_b"]
    control.append(["Cache-bridging, correct image"]
                   + [cell(runs[k], "accuracy") for k in keys])
    control.append(["Cache-bridging, mismatched image"]
                   + [cell(runs[k], "accuracy_shuffled_images") for k in keys])
    control.append(["  paired \u0394 vs mismatched, 95% CI"]
                   + [paired(runs[k], "paired_delta_shuffled") for k in keys])
    control.append(["Cache-bridging, visual cache disabled"]
                   + [cell(runs[k], "accuracy_no_visual") for k in keys])
    control.append(["  paired \u0394 vs disabled, 95% CI"]
                   + [paired(runs[k], "paired_delta_no_visual") for k in keys])
    keys_c = ["vqa_c", "ok_c", "gqa_c"]
    control.append(["Baseline, correct image"]
                   + [cell(runs[k], "accuracy") for k in keys_c])
    control.append(["Baseline, mismatched image"]
                   + [cell(runs[k], "accuracy_shuffled_images")
                      for k in keys_c])
    control.append(["  paired \u0394 vs mismatched, 95% CI"]
                   + [paired(runs[k], "paired_delta_shuffled")
                      for k in keys_c])
    control.append(["Baseline, visual prefix zeroed"]
                   + [cell(runs[k], "accuracy_no_visual") for k in keys_c])

    order = ["question only (blind control)",
             "cache-bridging, cache disabled",
             "caption-style projection",
             "cache-bridging, full cache",
             "cache-bridging, 50% top-K",
             "cache-bridging, 25% top-K"]
    labels = {"question only (blind control)": "Question only (blind control)",
              "cache-bridging, cache disabled":
                  "Cache-bridging, cache disabled",
              "caption-style projection": "Caption-style projection",
              "cache-bridging, full cache": "Cache-bridging, full cache",
              "cache-bridging, 50% top-K":
                  "Cache-bridging, Top-K keeping 50%",
              "cache-bridging, 25% top-K":
                  "Cache-bridging, Top-K keeping 25%"}
    by_system = dict((v["system"], v) for v in runs["cost"]["variants"])
    cost = [["System", "Latency, mean \u00b1 SD (ms)", "Median (ms)",
             "Peak memory (MiB)"]]
    for key in order:
        v = by_system[key]
        cost.append([labels[key],
                     "%.1f \u00b1 %.1f" % (v["latency_ms_mean"],
                                        v["latency_ms_std"]),
                     "%.1f" % v["latency_ms_median"],
                     "%.0f" % (v["peak_mem_bytes"] / 1024.0 ** 2)])

    base = abl["full"]["accuracy"]
    rows = [("full", "Full bridge, all four levels"),
            ("no_geometry", "without the 2D geometry bias"),
            ("no_gate", "without the gate (parameters removed with it)"),
            ("gate_fixed", "gate fixed to an even blend (matched params)"),
            ("per_layer_scale", "with a learned K/V scale per decoder layer"),
            ("drop_basic", "without the basic level"),
            ("drop_object", "without the semantic-object level"),
            ("drop_relation", "without the spatial-relation level"),
            ("drop_attribute", "without the attribute level"),
            ("only_basic", "basic level only"),
            ("only_object", "semantic-object level only"),
            ("only_relation", "spatial-relation level only"),
            ("only_attribute", "attribute level only"),
            ("keep50", "Top-K compression, keeping 50%"),
            ("keep25", "Top-K compression, keeping 25%")]
    ablation = [["Variant", "GQA-spatial (95% CI)", "\u0394",
                 "Trainable params"]]
    for tag, label in rows:
        run = abl[tag]
        d = run["accuracy"] - base
        ablation.append([label, acc_ci(run),
                         "%+.2f" % d if tag != "full" else "\u2014",
                         "{:,}".format(run["connector_params"])])

    capacity = [["Bridge configuration", "Trainable params",
                 "GQA-spatial (95% CI)"]]
    for tag, run in runs["capacity"]:
        cfg = run["config"]
        capacity.append(["rank %d, %d tokens per level"
                         % (cfg["rank"], cfg["n_visual"]),
                         "{:,}".format(run["connector_params"]),
                         acc_ci(run)])
    capacity.append(["Caption-style baseline, for reference",
                     "{:,}".format(runs["gqa_c"]["connector_params"]),
                     acc_ci(runs["gqa_c"])])

    prefix = [["Total prefix tokens", "Caption-style (95% CI)",
               "Cache-bridging (95% CI)", "Caption - bridge, paired 95% CI"]]
    for tag in ("6", "9", "18", "36"):
        caption_run = load("run_prefix%s_caption.json" % tag)
        bridge_run = load("run_prefix%s_bridge.json" % tag)
        difference = runs["paired"]["prefix%s_caption_minus_bridge" % tag]
        prefix.append([
            int(tag) * 4,
            acc_ci(caption_run),
            acc_ci(bridge_run),
            paired({"paired_delta_shuffled": difference},
                   "paired_delta_shuffled"),
        ])

    tables = {"levels": levels, "main": main, "control": control,
              "cost": cost, "ablation": ablation, "capacity": capacity,
              "prefix": prefix}

    probe = runs.get("probe_b")
    if probe is not None:
        mass = probe["attention_mass"]["per_layer_mean"]
        chance = probe["attention_mass"]["chance_level"]
        rows_a = [["Decoder layer", "Attention on visual prefix",
                   "Relative to chance"]]
        for idx, value in enumerate(mass):
            rows_a.append([idx, "%.4f" % value, "%.2fx" % (value / chance)])
        tables["attention"] = rows_a
    return tables


TABLE_CAPTIONS = {
    "levels": ("Table 1. The four cache levels and the CLIP-ViT-L/14 encoder "
               "layer each is tapped from, with n = {clip_layers}. The content "
               "column states the hypothesis behind the design, taken from the "
               "layer-wise probing literature @@CITE:dosovitskiy2021vit@@; it "
               "is not a result of this paper, and Section 4.5 tests it."),
    "main": ("Table 2. Accuracy (%) with 95% bootstrap confidence "
             "intervals on {eval_size} held-out questions per dataset, after "
             "{steps} steps under an identical training protocol with both "
             "backbones frozen. The blind control never reads the image. "
             "OK-VQA uses ten-reference VQA-style soft scoring; VQAv2 and GQA "
             "are normalised exact match with the references available in "
             "Section 3.5."),
    "control": ("Table 3. Control conditions (%). \u201cMismatched image\u201d scores "
                "the same trained weights with every question paired with "
                "another question\u2019s image; \u201cvisual cache disabled\u201d zeroes "
                "the level mask and \u201cvisual prefix zeroed\u201d the baseline\u2019s "
                "soft prefix. Paired \u0394 rows are bootstrap intervals over the "
                "per-question differences; \u201cn.s.\u201d marks an interval that "
                "contains zero."),
    "cost": ("Table 4. Single-sample inference cost, {cost_repeats} questions, "
             "up to {cost_new_tokens} generated tokens, measured back to back in one "
             "process on one {gpu} with the trained GQA-spatial checkpoints. "
             "Peak memory is quoted in MiB because the "
             "two frozen backbones dominate it and the connectors differ by "
             "far less than a gigabyte."),
    "ablation": ("Table 5. Ablations on GQA-spatial at {abl_steps} steps, "
                 "{eval_size} held-out questions. Each variant changes one "
                 "component; \u0394 is relative to the full bridge. Intervals are "
                 "about \u00b1{abl_ci_half} points, so smaller deltas are not "
                 "interpretable individually."),
    "capacity": ("Table 6. Capacity sweep on GQA-spatial under the "
                 "main-comparison protocol. The bridge is grown over "
                 "{cap_range} trainable parameters, bracketing the "
                 "baseline\u2019s {caption_params}."),
    "prefix": ("Table 8. Matched total-prefix comparison on GQA-spatial at "
               "{prefix_steps} steps. Both systems receive the same total "
               "number of prefix tokens in each row. The final column is a "
               "paired bootstrap over per-question score differences."),
    "attention": ("Table 7. Share of text-token attention probability that "
                  "lands on the injected visual prefix, per decoder layer, "
                  "averaged over heads and over {probe_tokens} scored text "
                  "tokens. Chance for a uniform reader is {probe_chance}."),
}


TABLE_STYLES = {"levels": "Light Grid", "main": "Grid Table 6 Colorful",
                "control": "Grid Table 6 Colorful",
                "cost": "Grid Table 6 Colorful",
                "ablation": "Grid Table 6 Colorful",
                "capacity": "Grid Table 6 Colorful",
                "prefix": "Grid Table 6 Colorful",
                "attention": "Grid Table 6 Colorful"}
