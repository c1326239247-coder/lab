# -*- coding: utf-8 -*-
"""Collect every number the manuscript quotes from results/*.json.

Nothing in the manuscript is typed by hand: build_docx.py substitutes the
values produced here, and an unresolved placeholder aborts the build, so
the text cannot drift away from the measurements in the repository.
"""

import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load(name):
    path = os.path.join(RESULTS, name)
    with io.open(path, encoding="utf-8") as handle:
        return json.load(handle)


def maybe(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    return load(name)


def pct(value):
    return "%.2f%%" % value


def ci(run, key="accuracy_ci95"):
    """A mean with its 95% bootstrap interval, as the text quotes it."""
    block = run[key]
    return "%.2f%% (95%% CI %.2f-%.2f)" % (block["mean"], block["low"],
                                           block["high"])


def half(run, key="accuracy_ci95"):
    return "%.2f" % run[key]["half_width"]


def delta(block):
    """A paired difference with its interval and verdict."""
    return "%+.2f points (95%% CI %+.2f to %+.2f)" % (
        block["delta"], block["low"], block["high"])


def paired_sentence(block, left, right):
    """Describe a paired difference without inferring past its interval."""
    verdict = ("excludes zero" if block["significant"] else "includes zero")
    return ("The paired bootstrap for %s minus %s gives %s; the interval %s."
            % (left, right, delta(block), verdict))


def mean_sd(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var ** 0.5


def facts():
    gqa_b = load("run_GQA_bridge.json")
    gqa_c = load("run_GQA_caption.json")
    gqa_n = load("run_GQA_blind.json")
    vqa_b = load("run_VQAv2_bridge.json")
    vqa_c = load("run_VQAv2_caption.json")
    vqa_n = load("run_VQAv2_blind.json")
    ok_b = load("run_OKVQA_bridge.json")
    ok_c = load("run_OKVQA_caption.json")
    ok_n = load("run_OKVQA_blind.json")
    cost = load("inference_cost.json")
    back = load("backbone_params.json")
    paired = load("paired_comparisons.json")["comparisons"]
    split_audit = load("split_audit.json")

    abl_tags = ["full", "no_geometry", "no_gate", "gate_fixed",
                "per_layer_scale", "drop_basic", "drop_object",
                "drop_relation", "drop_attribute", "only_basic",
                "only_object", "only_relation", "only_attribute",
                "keep50", "keep25"]
    abl = dict((tag, load("run_abl_%s.json" % tag)) for tag in abl_tags)

    cap_tags = ["rank64", "rank128", "rank256", "rank512", "rank1024",
                "rank2304"]
    capacity = [(tag, load("run_cap_%s.json" % tag)) for tag in cap_tags]

    prefix_tags = ["6", "9", "18", "36"]
    prefix_b = {tag: load("run_prefix%s_bridge.json" % tag)
                for tag in prefix_tags}
    prefix_c = {tag: load("run_prefix%s_caption.json" % tag)
                for tag in prefix_tags}
    prefix_sentence = "; ".join(
        "%d tokens: caption %s against bridge %s"
        % (int(tag) * 4, pct(prefix_c[tag]["accuracy"]),
           pct(prefix_b[tag]["accuracy"]))
        for tag in prefix_tags)
    prefix_deltas = [prefix_b[tag]["accuracy"] - prefix_c[tag]["accuracy"]
                     for tag in prefix_tags]
    if all(d < 0 for d in prefix_deltas):
        prefix_verdict = (
            "The caption baseline has the higher point estimate at every "
            "matched length."
        )
    elif all(d > 0 for d in prefix_deltas):
        prefix_verdict = (
            "The bridge has the higher point estimate at every matched length."
        )
    elif all(abs(d) < 1e-12 for d in prefix_deltas):
        prefix_verdict = "The two systems tie by point estimate at every length."
    else:
        prefix_verdict = (
            "The point-estimate ordering changes with prefix length, so the "
            "two factors cannot be interpreted independently in this sweep."
        )

    kv_tags = ["05", "10", "20"]
    kv_runs = {tag: load("run_kvscale%s.json" % tag) for tag in kv_tags}
    kv_probes = {tag: maybe("probe_kvscale%s.json" % tag) for tag in kv_tags}
    kv_sentence = "; ".join(
        "scale %s: %s, attention %.3f" % (
            "0.5" if tag == "05" else ("1.0" if tag == "10" else "2.0"),
            pct(kv_runs[tag]["accuracy"]),
            (kv_probes[tag]["attention_mass"]["overall_mean"]
             if kv_probes[tag] is not None else float("nan")))
        for tag in kv_tags)
    kv_half_better = (kv_runs["05"]["accuracy"]
                      > kv_runs["10"]["accuracy"])
    kv_double_better = (kv_runs["20"]["accuracy"]
                        > kv_runs["10"]["accuracy"])
    if kv_half_better and not kv_double_better:
        kv_verdict = (
            "In these separately retrained runs, reducing the scale from 1.0 "
            "to 0.5 coincides with higher accuracy and lower prefix attention, "
            "whereas doubling it does not improve accuracy."
        )
    else:
        kv_verdict = (
            "Accuracy and prefix attention do not follow the simple monotonic "
            "pattern predicted by the original crowding hypothesis."
        )

    mistral_b = load("run_Mistral7B_GQA_blind.json")
    mistral_c = load("run_Mistral7B_GQA_caption.json")
    mistral_g = load("run_Mistral7B_GQA_bridge.json")
    mistral_novis_change = (mistral_g["accuracy_no_visual"]
                            - mistral_g["accuracy"])
    if abs(mistral_novis_change) < 1e-12:
        mistral_novis_clause = ("leaves accuracy unchanged at %s"
                                % pct(mistral_g["accuracy_no_visual"]))
    else:
        mistral_novis_clause = "%s it to %s" % (
            "raises" if mistral_novis_change > 0 else "lowers",
            pct(mistral_g["accuracy_no_visual"]))
    mistral_sentence = (
        "On the larger Mistral-7B-Instruct-v0.2 decoder used in 4-bit precision "
        "with the same 3,000/2,000 GQA-spatial protocol, the blind control "
        "reaches %s, the caption-style baseline %s, and cache-bridging %s; "
        "disabling the bridge's visual pathway %s."
        % (ci(mistral_b), ci(mistral_c), ci(mistral_g),
           mistral_novis_clause))
    if mistral_c["accuracy"] > mistral_g["accuracy"]:
        mistral_verdict = (
            "The caption connector remains ahead by point estimate in this "
            "limited check."
        )
    elif mistral_c["accuracy"] < mistral_g["accuracy"]:
        mistral_verdict = (
            "The bridge is ahead by point estimate in this limited check, so "
            "the TinyLlama ordering does not generalise unchanged."
        )
    else:
        mistral_verdict = "The two visual connectors tie in this limited check."
    mistral_grounding = paired_sentence(
        mistral_g["paired_delta_shuffled"], "the correct image",
        "the mismatched image")
    if mistral_g["paired_delta_shuffled"]["significant"]:
        mistral_grounding += " Image-dependent signal is therefore detectable in this run."
    else:
        mistral_grounding += (" This run does not establish image-dependent "
                              "signal at 7B scale.")

    probe_b = maybe("probe_GQA_bridge.json")
    probe_c = maybe("probe_GQA_caption.json")

    cfg = gqa_b["config"]
    dev = gqa_b["device"]
    by_system = dict((v["system"], v) for v in cost["variants"])
    cap_cost = by_system["caption-style projection"]
    brg_cost = by_system["cache-bridging, full cache"]
    top50_cost = by_system["cache-bridging, 50% top-K"]
    top25_cost = by_system["cache-bridging, 25% top-K"]
    latency_change = (100.0 * (brg_cost["latency_ms_mean"]
                              - cap_cost["latency_ms_mean"])
                      / cap_cost["latency_ms_mean"])
    if latency_change >= 0:
        latency_sentence = (
            "Injecting the cache costs %.1f%% more latency than the "
            "caption-style connector" % latency_change)
    else:
        latency_sentence = (
            "Injecting the cache reduces mean latency by %.1f%% relative to "
            "the caption-style connector" % -latency_change)
    latency_sentence += (" (%.1f \u00b1 %.1f ms against %.1f \u00b1 %.1f ms)."
                         % (brg_cost["latency_ms_mean"],
                            brg_cost["latency_ms_std"],
                            cap_cost["latency_ms_mean"],
                            cap_cost["latency_ms_std"]))
    topk_sentence = (
        "Mean latency is %.1f ms for the full cache, %.1f ms for Top-K 50%%, "
        "and %.1f ms for Top-K 25%%. These measurements include the selection "
        "step and should be interpreted as implementation costs rather than "
        "asymptotic scaling."
        % (brg_cost["latency_ms_mean"], top50_cost["latency_ms_mean"],
           top25_cost["latency_ms_mean"]))

    n_visual = cfg["n_visual"]
    side = int(round(n_visual ** 0.5))
    total_backbone = back["clip_params"] + back["llm_params"]
    small = gqa_b["connector_params"]

    abl_full = abl["full"]["accuracy"]
    drop = dict((k, abl["drop_" + k]["accuracy"] - abl_full)
                for k in ["basic", "object", "relation", "attribute"])
    only = dict((k, abl["only_" + k]["accuracy"])
                for k in ["basic", "object", "relation", "attribute"])

    # Seed repeats, reported for both systems under identical conditions.
    bridge_runs = [gqa_b["accuracy"]]
    caption_runs = [gqa_c["accuracy"]]
    seed_pairs = []
    for seed in [1, 2]:
        b = maybe("run_GQA_bridge_seed%d.json" % seed)
        c = maybe("run_GQA_caption_seed%d.json" % seed)
        if b is not None:
            bridge_runs.append(b["accuracy"])
        if c is not None:
            caption_runs.append(c["accuracy"])
        if b is not None and c is not None:
            seed_pairs.append((seed, c["accuracy"], b["accuracy"]))
    bridge_mean, bridge_sd = mean_sd(bridge_runs)
    caption_mean, caption_sd = mean_sd(caption_runs)
    if seed_pairs:
        parts = ["seed %d gives %s against %s" % (s, pct(c), pct(b))
                 for s, c, b in seed_pairs]
        seed_order = ("preserve the caption-over-bridge point-estimate ordering"
                      if all(c > b for _, c, b in seed_pairs)
                      else "do not preserve one point-estimate ordering")
        seed_sentence = ("Two further initialisations %s: %s for the baseline "
                         "and the bridge respectively. Over the %d seeds the "
                         "baseline averages %.2f%% (SD %.2f) and the bridge "
                         "%.2f%% (SD %.2f)."
                         % (seed_order, "; ".join(parts), len(caption_runs),
                            caption_mean, caption_sd, bridge_mean, bridge_sd))
    else:
        seed_sentence = ""

    # Capacity sweep: the largest bridge against the default one.
    cap_best_tag, cap_best = max(capacity,
                                 key=lambda item: item[1]["accuracy"])
    cap_largest = max(capacity,
                      key=lambda item: item[1]["connector_params"])[1]
    cap_range = "%s to %s" % ("{:,}".format(
        min(r["connector_params"] for _, r in capacity)),
        "{:,}".format(max(r["connector_params"] for _, r in capacity)))
    cap_span = max(r["accuracy"] for _, r in capacity) - \
        min(r["accuracy"] for _, r in capacity)
    cap_sentence = "; ".join(
        "%s parameters: %s" % ("{:,}".format(r["connector_params"]),
                               pct(r["accuracy"]))
        for _, r in capacity)
    cap_delta = cap_best["accuracy"] - gqa_c["accuracy"]
    if cap_delta > 0:
        capacity_verdict = (
            "The best bridge reaches %s at %s parameters, exceeding the "
            "caption baseline's %s by %.2f points. A bridge above the "
            "baseline's parameter budget reaches %s. Connector configuration "
            "therefore materially affects accuracy, and capacity cannot be "
            "dismissed as an explanation."
            % (pct(cap_best["accuracy"]),
               "{:,}".format(cap_best["connector_params"]),
               pct(gqa_c["accuracy"]), cap_delta, pct(cap_largest["accuracy"]))
        )
    elif cap_delta < 0:
        capacity_verdict = (
            "The best bridge reaches %s at %s parameters, remaining %.2f "
            "points below the caption baseline's %s. A bridge above the "
            "baseline's parameter budget reaches %s. Capacity affects the "
            "outcome but the best sampled configuration does not close the "
            "point-estimate gap."
            % (pct(cap_best["accuracy"]),
               "{:,}".format(cap_best["connector_params"]), -cap_delta,
               pct(gqa_c["accuracy"]), pct(cap_largest["accuracy"]))
        )
    else:
        capacity_verdict = (
            "The best sampled bridge ties the caption baseline at %s, so "
            "capacity cannot be dismissed as an explanation."
            % pct(cap_best["accuracy"])
        )

    # Sentences whose wording depends on the direction of a measured
    # difference are generated here rather than asserted in the body text,
    # so the manuscript stays correct if a rerun changes the ordering.
    def compare(name, bridge, blind, caption):
        b = bridge["accuracy"]
        n = blind["accuracy"]
        c = caption["accuracy"]
        if b > n:
            rel = "above the blind control by %.2f points" % (b - n)
        else:
            rel = "below the blind control by %.2f points" % (n - b)
        return "on %s the bridge is %s and %.2f points below the baseline" % (
            name, rel, c - b)

    blind_sentence = "; ".join([
        compare("VQAv2", vqa_b, vqa_n, vqa_c),
        compare("OK-VQA", ok_b, ok_n, ok_c),
        compare("GQA-spatial", gqa_b, gqa_n, gqa_c)]) + "."

    beats_blind = [name for name, b, n in
                   [("VQAv2", vqa_b, vqa_n), ("OK-VQA", ok_b, ok_n),
                    ("GQA-spatial", gqa_b, gqa_n)]
                   if b["accuracy"] > n["accuracy"]]
    if not beats_blind:
        blind_verdict = ("On none of the three datasets does the bridge reach "
                         "the accuracy of a connector that never sees the "
                         "image at all")
    elif len(beats_blind) == 3:
        blind_verdict = ("The bridge is ahead of the blind control on all "
                         "three datasets, so the visual pathway contributes "
                         "on every dataset")
    else:
        blind_verdict = ("The bridge is ahead of the blind control on %s only"
                         % " and ".join(beats_blind))

    # The no-visual control moves in different directions per dataset; the
    # text quotes this sentence instead of assuming a sign.
    def novis_dir(name, run):
        d = run["accuracy_no_visual"] - run["accuracy"]
        verb = "raises" if d > 0 else "lowers"
        return "%s: disabling the cache %s accuracy by %.2f points" % (
            name, verb, abs(d))

    novis_sentence = "; ".join([novis_dir("VQAv2", vqa_b),
                                novis_dir("OK-VQA", ok_b),
                                novis_dir("GQA-spatial", gqa_b)]) + "."

    # The share of held-out answers that are empty or a verbatim word of the
    # question, which is what the failure-case discussion quotes.
    def degenerate_share(run):
        rows = run.get("samples") or []
        if not rows:
            return None
        bad = 0
        for row in rows:
            prediction = (row.get("prediction") or "").strip().lower()
            question = (row.get("question") or "").lower()
            words = set(question.replace("?", " ").split())
            if not prediction or prediction in words:
                bad += 1
        return 100.0 * bad / len(rows)

    deg_bridge = degenerate_share(ok_b)
    deg_caption = degenerate_share(ok_c)

    # Whether the mismatched-image control is significant is itself a
    # per-dataset outcome, so the sentence that states it is generated.
    sig = [name for name, run in [("VQAv2", vqa_b), ("OK-VQA", ok_b),
                                  ("GQA-spatial", gqa_b)]
           if run["paired_delta_shuffled"]["significant"]]
    nonsig = [name for name, run in [("VQAv2", vqa_b), ("OK-VQA", ok_b),
                                     ("GQA-spatial", gqa_b)]
              if not run["paired_delta_shuffled"]["significant"]]
    if not nonsig:
        grounding_sentence = ("On all three datasets the paired interval "
                              "excludes zero, so the injected cache is "
                              "demonstrably image-dependent everywhere")
    elif not sig:
        grounding_sentence = ("On none of the three datasets does the paired "
                              "interval exclude zero, so we cannot establish "
                              "that the injected cache is image-dependent at "
                              "all")
    else:
        grounding_sentence = (
            "The paired interval excludes zero on %s but not on %s, so the "
            "injected cache is demonstrably image-dependent on the former and "
            "indistinguishable from an image-independent connector on the "
            "latter" % (" and ".join(sig), " and ".join(nonsig)))

    # The level ablations can go either way, so the sentence that reads them
    # is generated from the signs rather than assumed.
    level_names = {"basic": "basic", "object": "semantic-object",
                   "relation": "spatial-relation", "attribute": "attribute"}
    hurt = [(level_names[k], -v) for k, v in drop.items() if v < 0]
    helped = [(level_names[k], v) for k, v in drop.items() if v >= 0]
    parts = []
    if hurt:
        parts.append("removing the %s level%s costs %s point%s" % (
            " and the ".join(n for n, _ in hurt),
            "" if len(hurt) == 1 else "s",
            " and ".join("%.2f" % d for _, d in hurt),
            "" if len(hurt) == 1 and abs(hurt[0][1] - 1) < 1e-9 else "s"))
    if helped:
        parts.append("removing the %s level%s in fact improves accuracy by %s "
                     "point%s" % (
                         " and the ".join(n for n, _ in helped),
                         "" if len(helped) == 1 else "s",
                         " and ".join("%.2f" % d for _, d in helped),
                         "" if len(helped) == 1 else "s"))
    drop_sentence = ", while ".join(parts)

    best_single = max(only.values())
    if best_single >= abl_full:
        single_verdict = ("the best single-level variant, %s, already matches "
                          "the full hierarchy at %s, so the hierarchy is not "
                          "earning its four-fold token cost"
                          % (pct(best_single), pct(abl_full)))
    else:
        single_verdict = ("no single level matches the full hierarchy \u2014 the "
                          "best single-level variant is %s against %s"
                          % (pct(best_single), pct(abl_full)))

    ordered = sorted(only.items(), key=lambda kv: -kv[1])
    if ordered[0][0] == "relation":
        level_hypothesis = ("the level named for spatial relations is indeed "
                            "the strongest in isolation on the "
                            "relation-heavy dataset, which is consistent with "
                            "the hypothesis of Table 1")
    else:
        level_hypothesis = ("the strongest level in isolation is the one named "
                            "%s rather than the one named for spatial "
                            "relations, which the hypothesis of Table 1 would "
                            "have predicted on a relation-heavy dataset"
                            % level_names[ordered[0][0]])

    # The gate ablation has two variants and their ordering is a result, not
    # an assumption, so the sentence that interprets them is generated.
    gate_fixed_delta = abl_full - abl["gate_fixed"]["accuracy"]
    gate_half = abl["full"]["accuracy_ci95"]["half_width"]
    if abs(gate_fixed_delta) <= gate_half:
        gate_verdict = (
            "It reaches %s, close to the full bridge under this single-run "
            "measurement, so the results do not support a necessary role for "
            "the adaptive coefficient"
            % pct(abl["gate_fixed"]["accuracy"]))
    elif gate_fixed_delta > 0:
        gate_verdict = (
            "It reaches %s, %.2f points below the full bridge. This is "
            "consistent with a benefit from adaptivity in this run, but the "
            "unrepeated comparison does not establish that benefit"
            % (pct(abl["gate_fixed"]["accuracy"]), gate_fixed_delta))
    else:
        gate_verdict = (
            "It reaches %s, which is %.2f points above the full bridge, so the "
            "learned coefficient is not helping in this run"
            % (pct(abl["gate_fixed"]["accuracy"]), -gate_fixed_delta))

    geo_delta = abl_full - abl["no_geometry"]["accuracy"]
    if abs(geo_delta) <= gate_half:
        geo_verdict = ("removing the 2D geometry bias changes accuracy by "
                       "%.2f points at an essentially unchanged parameter "
                       "count, which is inside the interval and therefore not "
                       "evidence either way" % abs(geo_delta))
    elif geo_delta > 0:
        geo_verdict = ("removing the 2D geometry bias costs %.2f points at an "
                       "essentially unchanged parameter count" % geo_delta)
    else:
        geo_verdict = ("removing the 2D geometry bias in fact improves "
                       "accuracy by %.2f points at an essentially unchanged "
                       "parameter count, so the bias is not earning its place "
                       "at this budget" % (-geo_delta))

    pls = abl["per_layer_scale"]
    pls_delta = pls["accuracy"] - abl_full
    pls_block = pls["accuracy_ci95"]
    if abs(pls_delta) <= pls_block["half_width"]:
        pls_verdict = ("The difference is inside the interval of the "
                       "measurement, so giving each layer its own scale does "
                       "not recover the deficit: sharing one cache across the "
                       "depth is not, on its own, the binding constraint")
    elif pls_delta > 0:
        pls_verdict = ("The gain exceeds the interval of the measurement, so "
                       "part of the deficit is attributable to feeding every "
                       "layer the same cache")
    else:
        pls_verdict = ("The variant is worse than the shared cache by more "
                       "than the interval of the measurement, so per-layer "
                       "freedom costs rather than helps at this budget")

    total_seconds = 0.0
    for name in os.listdir(RESULTS):
        if (name.startswith("run_") and not name.startswith("run_smoke")
                and name.endswith(".json")):
            run = load(name)
            # eval_seconds covers the correct-image pass. Visual systems are
            # also evaluated with mismatched images and a disabled pathway;
            # blind controls reuse the same scores for those two conditions.
            eval_passes = 1 if run.get("system") == "blind" else 3
            total_seconds += (run.get("train_seconds", 0)
                              + eval_passes * run.get("eval_seconds", 0))

    values = {
        # --- main comparison
        "gqa_bridge": pct(gqa_b["accuracy"]),
        "gqa_bridge_ci": ci(gqa_b),
        "gqa_caption": pct(gqa_c["accuracy"]),
        "gqa_caption_ci": ci(gqa_c),
        "gqa_blind": pct(gqa_n["accuracy"]),
        "gqa_blind_ci": ci(gqa_n),
        "gqa_shuf": pct(gqa_b["accuracy_shuffled_images"]),
        "gqa_novis": pct(gqa_b["accuracy_no_visual"]),
        "gqa_caption_shuf": pct(gqa_c["accuracy_shuffled_images"]),
        "gqa_caption_novis": pct(gqa_c["accuracy_no_visual"]),
        "vqa_bridge": pct(vqa_b["accuracy"]),
        "vqa_caption": pct(vqa_c["accuracy"]),
        "vqa_blind": pct(vqa_n["accuracy"]),
        "vqa_shuf": pct(vqa_b["accuracy_shuffled_images"]),
        "vqa_novis": pct(vqa_b["accuracy_no_visual"]),
        "ok_bridge": pct(ok_b["accuracy"]),
        "ok_caption": pct(ok_c["accuracy"]),
        "ok_blind": pct(ok_n["accuracy"]),
        "ok_shuf": pct(ok_b["accuracy_shuffled_images"]),
        "ok_novis": pct(ok_b["accuracy_no_visual"]),
        "ci_half_typical": half(gqa_b),
        "ci_half_caption": half(gqa_c),
        # --- gaps, all as paired quantities where the pairing exists
        "gap_gqa": "%.2f" % (gqa_c["accuracy"] - gqa_b["accuracy"]),
        "gap_vqav2": "%.2f" % (vqa_c["accuracy"] - vqa_b["accuracy"]),
        "gap_okvqa": "%.2f" % (ok_c["accuracy"] - ok_b["accuracy"]),
        "gap_blind_gqa": "%.2f" % (gqa_n["accuracy"] - gqa_b["accuracy"]),
        "gap_blind_vqa": "%.2f" % (vqa_n["accuracy"] - vqa_b["accuracy"]),
        "gap_blind_ok": "%.2f" % (ok_n["accuracy"] - ok_b["accuracy"]),
        "delta_gqa_shuf": delta(gqa_b["paired_delta_shuffled"]),
        "delta_gqa_novis": delta(gqa_b["paired_delta_no_visual"]),
        "delta_vqa_shuf": delta(vqa_b["paired_delta_shuffled"]),
        "delta_vqa_novis": delta(vqa_b["paired_delta_no_visual"]),
        "delta_ok_shuf": delta(ok_b["paired_delta_shuffled"]),
        "delta_caption_shuf": delta(gqa_c["paired_delta_shuffled"]),
        # --- seeds and spread
        "seed_sentence": seed_sentence,
        "blind_sentence": blind_sentence,
        "gqa_main_paired": paired_sentence(
            paired["GQA_caption_minus_bridge"], "caption", "bridge"),
        "vqa_main_paired": paired_sentence(
            paired["VQAv2_caption_minus_bridge"], "caption", "bridge"),
        "ok_main_paired": paired_sentence(
            paired["OKVQA_caption_minus_bridge"], "caption", "bridge"),
        "capacity_paired": paired_sentence(
            paired["capacity_best_minus_caption"], "the best bridge",
            "caption"),
        "mistral_paired": paired_sentence(
            paired["Mistral7B_caption_minus_bridge"], "caption", "bridge"),
        "blind_verdict": blind_verdict,
        "novis_sentence": novis_sentence,
        "grounding_sentence": grounding_sentence,
        "deg_bridge": "\u2014" if deg_bridge is None else "%.0f%%" % deg_bridge,
        "deg_caption": ("\u2014" if deg_caption is None
                        else "%.0f%%" % deg_caption),
        "n_samples": "%d" % len(ok_b.get("samples") or []),
        "pls_verdict": pls_verdict,
        "gate_verdict": gate_verdict,
        "drop_sentence": drop_sentence,
        "single_verdict": single_verdict,
        "level_hypothesis": level_hypothesis,
        "geo_verdict": geo_verdict,
        "bridge_mean": pct(bridge_mean),
        "bridge_sd": "%.2f" % bridge_sd,
        "caption_mean": pct(caption_mean),
        "caption_sd": "%.2f" % caption_sd,
        "bridge_max": pct(max(bridge_runs)),
        "caption_min": pct(min(caption_runs)),
        "worst_case_gap": "%.2f" % (min(caption_runs) - max(bridge_runs)),
        "n_bridge_runs": "%d" % len(bridge_runs),
        "n_caption_runs": "%d" % len(caption_runs),
        # --- capacity sweep
        "cap_sentence": cap_sentence,
        "cap_range": cap_range,
        "cap_span": "%.2f" % cap_span,
        "cap_best": pct(cap_best["accuracy"]),
        "cap_best_params": "{:,}".format(cap_best["connector_params"]),
        "cap_largest": pct(cap_largest["accuracy"]),
        "cap_largest_params": "{:,}".format(cap_largest["connector_params"]),
        "cap_n_points": "%d" % len(capacity),
        "capacity_verdict": capacity_verdict,
        "prefix_sentence": prefix_sentence,
        "prefix_verdict": prefix_verdict,
        "prefix_steps": prefix_b["36"]["config"]["steps"],
        "kv_sentence": kv_sentence,
        "kv_verdict": kv_verdict,
        "mistral_sentence": mistral_sentence,
        "mistral_verdict": mistral_verdict,
        "mistral_grounding": mistral_grounding,
        "mistral_caption_params": "{:,}".format(mistral_c["connector_params"]),
        "mistral_bridge_params": "{:,}".format(mistral_g["connector_params"]),
        # --- parameters and shapes
        "small_params": "{:,}".format(small),
        "caption_params": "{:,}".format(gqa_c["connector_params"]),
        "blind_params": "{:,}".format(gqa_n["connector_params"]),
        "k_std": "%.2f" % gqa_b["host_kv_stats"]["k_std"],
        "v_std": "%.2f" % gqa_b["host_kv_stats"]["v_std"],
        "kv_ratio": "%d" % round(gqa_b["host_kv_stats"]["k_std"]
                                 / gqa_b["host_kv_stats"]["v_std"]),
        "gpu": dev["gpu"],
        "vram": dev["vram_gb"],
        "cc": dev["compute_capability"],
        "torch_ver": dev["torch"],
        "cuda_ver": dev["cuda"],
        "clip_params": "{:,}".format(back["clip_params"]),
        "llm_params": "{:,}".format(back["llm_params"]),
        "llm_layers": back["llm_num_layers"],
        "llm_hidden": back["llm_hidden_size"],
        "llm_heads": back["llm_num_attention_heads"],
        "llm_kv_heads": back["llm_num_key_value_heads"],
        "llm_head_dim": back["llm_hidden_size"] // back["llm_num_attention_heads"],
        "llm_kv_dim": (back["llm_num_key_value_heads"]
                       * (back["llm_hidden_size"]
                          // back["llm_num_attention_heads"])),
        "clip_layers": back["clip_num_layers"],
        "vision_dim": back["clip_hidden_size"],
        "grid": back["clip_image_size"] // back["clip_patch_size"],
        "n_patches": (back["clip_image_size"] // back["clip_patch_size"]) ** 2,
        "n_visual": n_visual,
        "pool_side": side,
        "n_visual_total": n_visual * 4,
        "rank": cfg["rank"],
        "max_offset": side - 1,
        "total_backbone": "{:,}".format(total_backbone),
        "bridge_pct": "%.3f%%" % (100.0 * small / total_backbone),
        "summary_proj_params": "{:,}".format(back["llm_hidden_size"]
                                            * (back["llm_num_key_value_heads"]
                                               * (back["llm_hidden_size"]
                                                  // back["llm_num_attention_heads"]))),
        # --- protocol
        "lr": "%g" % cfg["lr"],
        "batch": cfg["batch_size"],
        "steps": cfg["steps"],
        "abl_steps": abl["full"]["config"]["steps"],
        "eval_size": cfg["eval_size"],
        "train_size": cfg["train_size"],
        "shared_images_vqa": split_audit["VQAv2"]["shared_images"],
        "shared_images_ok": split_audit["OK-VQA"]["shared_images"],
        "shared_images_gqa": split_audit["GQA-spatial"]["shared_images"],
        "eos_weight": "0.2",
        "max_new_tokens": 6,
        "n_boot": "10,000",
        # --- cost
        "cost_new_tokens": 6,
        "cost_repeats": brg_cost["n_measured"],
        "caption_latency": "%.1f" % cap_cost["latency_ms_mean"],
        "caption_latency_sd": "%.1f" % cap_cost["latency_ms_std"],
        "bridge_latency": "%.1f" % brg_cost["latency_ms_mean"],
        "bridge_latency_sd": "%.1f" % brg_cost["latency_ms_std"],
        "latency_pct": "%.1f%%" % (100.0 * (brg_cost["latency_ms_mean"]
                                            - cap_cost["latency_ms_mean"])
                                   / cap_cost["latency_ms_mean"]),
        "latency_sentence": latency_sentence,
        "topk_sentence": topk_sentence,
        "caption_mem_mb": "%.0f" % (cap_cost["peak_mem_bytes"] / 1024.0 ** 2),
        "bridge_mem_mb": "%.0f" % (brg_cost["peak_mem_bytes"] / 1024.0 ** 2),
        "bridge_train_s": "%.1f" % gqa_b["train_seconds"],
        "caption_train_s": "%.1f" % gqa_c["train_seconds"],
        "bridge_train_mem": "%.2f" % gqa_b["train_peak_mem_gb"],
        "caption_train_mem": "%.2f" % gqa_c["train_peak_mem_gb"],
        "bridge_train_s_min": "%.0f" % (gqa_b["train_seconds"] / 60.0),
        "total_gpu_hours": "%.1f" % (total_seconds / 3600.0),
        # --- losses
        "bridge_loss_first": "%.2f" % gqa_b["loss_first_25"],
        "bridge_loss_last": "%.2f" % gqa_b["loss_last_25"],
        "ok_loss_first": "%.2f" % ok_b["loss_first_25"],
        "ok_loss_last": "%.2f" % ok_b["loss_last_25"],
        "caption_loss_first": "%.2f" % gqa_c["loss_first_25"],
        "caption_loss_last": "%.2f" % gqa_c["loss_last_25"],
        # --- ablations
        "abl_full": pct(abl_full),
        "abl_gate": pct(abl["no_gate"]["accuracy"]),
        "abl_gate_delta": "%.2f" % (abl_full - abl["no_gate"]["accuracy"]),
        "abl_gate_params": "{:,}".format(abl["no_gate"]["connector_params"]),
        "abl_gate_fixed": pct(abl["gate_fixed"]["accuracy"]),
        "abl_gate_fixed_params": "{:,}".format(
            abl["gate_fixed"]["connector_params"]),
        "abl_gate_fixed_delta": "%.2f" % (abl_full
                                          - abl["gate_fixed"]["accuracy"]),
        "abl_pls": pct(abl["per_layer_scale"]["accuracy"]),
        "abl_pls_delta": "%+.2f" % (abl["per_layer_scale"]["accuracy"]
                                    - abl_full),
        "abl_pls_params": "{:,}".format(
            abl["per_layer_scale"]["connector_params"]),
        "abl_novis": pct(abl["full"]["accuracy_no_visual"]),
        "abl_geo": pct(abl["no_geometry"]["accuracy"]),
        "abl_geo_delta": "%.2f" % (abl_full - abl["no_geometry"]["accuracy"]),
        "abl_keep50": pct(abl["keep50"]["accuracy"]),
        "abl_keep25": pct(abl["keep25"]["accuracy"]),
        "abl_keep50_delta": "%.2f" % (abl_full - abl["keep50"]["accuracy"]),
        "abl_keep25_delta": "%.2f" % (abl_full - abl["keep25"]["accuracy"]),
        "abl_best_single": pct(max(only.values())),
        "abl_drop_min": "%.2f" % min(abs(v) for v in drop.values()),
        "abl_drop_max": "%.2f" % max(abs(v) for v in drop.values()),
        "abl_ci_half": half(abl["full"]),
        "only_basic": pct(only["basic"]),
        "only_object": pct(only["object"]),
        "only_relation": pct(only["relation"]),
        "only_attribute": pct(only["attribute"]),
        "drop_basic_delta": "%.2f" % abs(drop["basic"]),
        "drop_object_delta": "%.2f" % abs(drop["object"]),
        "drop_relation_delta": "%.2f" % abs(drop["relation"]),
        "drop_attribute_delta": "%.2f" % abs(drop["attribute"]),
        # --- tests
        "n_shape_tests": 7,
        "n_torch_tests": 10,
        "n_prompt_tests": 2,
        "n_numpy_tests": 9,
    }

    if probe_b is not None:
        mass = probe_b["attention_mass"]
        values["probe_mass_first_half"] = "%.3f" % (
            sum(mass["per_layer_mean"][:len(mass["per_layer_mean"]) // 2])
            / (len(mass["per_layer_mean"]) // 2))
        values["probe_mass_second_half"] = "%.3f" % (
            sum(mass["per_layer_mean"][len(mass["per_layer_mean"]) // 2:])
            / (len(mass["per_layer_mean"])
               - len(mass["per_layer_mean"]) // 2))
        per_layer = mass["per_layer_mean"]
        values.update({
            "probe_mass": "%.3f" % mass["overall_mean"],
            "probe_chance": "%.3f" % mass["chance_level"],
            "probe_ratio": "%.2f" % (mass["overall_mean"]
                                     / mass["chance_level"]),
            "probe_min_layer": "%.3f" % min(per_layer),
            "probe_max_layer": "%.3f" % max(per_layer),
            "probe_first_layer": "%.3f" % per_layer[0],
            "probe_last_layer": "%.3f" % per_layer[-1],
            "probe_prefix_len": mass["n_prefix"],
            "probe_tokens": "{:,}".format(mass["text_tokens_scored"]),
        })
    if probe_c is not None:
        mass_c = probe_c["attention_mass"]
        values.update({
            "probe_caption_mass": "%.3f" % mass_c["overall_mean"],
            "probe_caption_chance": "%.3f" % mass_c["chance_level"],
            "probe_caption_ratio": "%.2f" % (mass_c["overall_mean"]
                                             / mass_c["chance_level"]),
        })
    if probe_b is not None and probe_c is not None:
        mb = probe_b["attention_mass"]
        mc = probe_c["attention_mass"]
        values["probe_mass_gap"] = "%.3f" % abs(mc["overall_mean"]
                                                - mb["overall_mean"])
        # The two prefixes differ in length, so the interpretable quantity is
        # each one's mass relative to what a uniform reader would give it.
        rb = mb["overall_mean"] / mb["chance_level"]
        rc = mc["overall_mean"] / mc["chance_level"]
        if rb > 1.0 and rc < 1.0:
            values["probe_verdict"] = (
                "the injected cache takes %.2f times the attention a uniform "
                "reader would give a prefix of its length, while the "
                "embedding-space prefix takes only %.2f times its own share. "
                "The frozen decoder therefore does not ignore the injected "
                "cache, while the baseline prefix receives less than its own "
                "uniform share"
                % (rb, rc))
        elif rb < rc:
            values["probe_verdict"] = (
                "the injected cache takes %.2f times a uniform reader\u2019s share "
                "against %.2f for the embedding-space prefix, so the frozen "
                "decoder consults the injected cache less readily than it "
                "consults a soft prompt" % (rb, rc))
        else:
            values["probe_verdict"] = (
                "the injected cache takes %.2f times a uniform reader\u2019s share "
                "against %.2f for the embedding-space prefix, so lack of "
                "attention to the injected cache is not the explanation"
                % (rb, rc))
        values["probe_bridge_len"] = mb["n_prefix"]
        values["probe_caption_len"] = mc["n_prefix"]

    runs = {"gqa_b": gqa_b, "gqa_c": gqa_c, "gqa_n": gqa_n,
            "vqa_b": vqa_b, "vqa_c": vqa_c, "vqa_n": vqa_n,
            "ok_b": ok_b, "ok_c": ok_c, "ok_n": ok_n,
            "cost": cost, "abl": abl, "back": back,
            "capacity": capacity, "seeds": seed_pairs,
            "probe_b": probe_b, "probe_c": probe_c, "paired": paired}
    return values, runs


if __name__ == "__main__":
    values, _ = facts()
    for key in sorted(values):
        print("%-24s %s" % (key, values[key]))
