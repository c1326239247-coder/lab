# -*- coding: utf-8 -*-
"""Build the point-by-point reviewer response from the final run JSON."""

import io
import os
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numbers as facts_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "dist")
STEM = os.path.join(OUT_DIR, "Response_to_Reviewers_2026-10-30")


def items():
    """Return the response structure with values filled from raw results."""
    values, _ = facts_mod.facts()
    title = ("Response to the editor and reviewers\n\nManuscript: Cache "
             "bridging of hierarchical visual key values does not "
             "consistently improve visual question answering with a frozen "
             "language model")
    body = [
        ("P", "We thank the editor and reviewers for identifying the "
              "implementation and comparison problems. We corrected the "
              "prompt-summary leakage, propagated the configured baseline "
              "prefix length, reran the affected experiments, added paired "
              "system comparisons, and revised the claims that the new "
              "measurements do not support. All values below come from the "
              "corrected result files."),
        ("H1", "Editor major comments"),
        ("H2", "1. Possible train-inference leakage"),
        ("P", "Response: Confirmed and corrected. The previous training path "
              "averaged embeddings over the question and gold answer. The "
              "gate now uses a mask-weighted mean over question tokens only. "
              "Answer tokens and training right padding are excluded, and "
              "inference uses the same mask to exclude left padding. A "
              "regression check gives identical summaries for the same prompt "
              "in the training and inference paths."),
        ("P", "Changes and new results: Sections 3.5 and 5.2 now specify the "
              "mask. We reran the bridge main runs, bridge seeds, ablations, "
              "capacity and prefix sweeps, joint K/V scaling, Mistral bridge, "
              "and attention probes. The corrected bridge scores {vqa_bridge} "
              "on VQAv2, {ok_bridge} on OK-VQA, and {gqa_bridge} on "
              "GQA-spatial. The corresponding caption scores are "
              "{vqa_caption}, {ok_caption}, and {gqa_caption}; the blind "
              "scores are {vqa_blind}, {ok_blind}, and {gqa_blind}. Across "
              "three GQA-spatial seeds, caption averages {caption_mean} "
              "(SD {caption_sd}) and the bridge {bridge_mean} "
              "(SD {bridge_sd})."),
        ("H2", "2. Caption baseline prefix length and capacity bracket"),
        ("P", "Response: Both concerns were correct. build_system now "
              "passes cfg.n_visual to the caption and blind systems. The "
              "main comparison uses 36 tokens for caption and blind, and four "
              "levels of 36 tokens (144 total) for the bridge. Section 4.1 "
              "states this asymmetry explicitly; Section 4.8 reports matched "
              "total lengths of 24, 36, 72, and 144 tokens with "
              "{prefix_steps} steps per run. The matched-length "
              "results are {prefix_sentence}. {prefix_verdict}"),
        ("P", "The capacity sweep now includes 6,556,451 trainable parameters, "
              "above the caption baseline's 6,299,648. The complete series is "
              "{cap_sentence}. {capacity_verdict} {capacity_paired} We have "
              "withdrawn the claim that capacity cannot explain the gap. "
              "Sections 4.6 and 6 also state that rank and token count vary "
              "together and that the best row is selected on the reported set "
              "without seed repetition."),
        ("H2", "3. Evidence for attention crowding"),
        ("P", "Response: We added the matched-prefix controls above and a "
              "joint K/V scale sweep at 0.5, 1.0, and 2.0. The results are "
              "{kv_sentence}. {kv_verdict} Sections 4.7, 4.8, 5.3, and 6 "
              "now give a narrower interpretation: scaling both K and V "
              "changes attention logits and value magnitudes together, so "
              "the experiment does not isolate attention crowding as the "
              "single cause."),
        ("P", "We also cite DynamicKV with the verified DOI "
              "10.18653/v1/2025.findings-emnlp.426 and discuss layer-adaptive "
              "retention as an untested direction, rather than presenting our "
              "global scale as a per-layer retention experiment."),
        ("H2", "4. Language-model scale"),
        ("P", "Response: We added a 4-bit Mistral-7B-Instruct-v0.2 sensitivity "
              "check on GQA-spatial with the same 3,000/2,000 protocol. "
              "{mistral_sentence} {mistral_verdict} {mistral_paired} "
              "{mistral_grounding} The "
              "caption connector has {mistral_caption_params} trainable "
              "parameters against {mistral_bridge_params} for the bridge. "
              "Section 4.9 therefore describes this as a single-dataset, "
              "non-capacity-matched check, not a full 7B replication or proof "
              "that frozen read-out paths inherently fail."),
        ("H2", "5. Presentation and title"),
        ("P", "Response: The title is now one sentence without a colon. The "
              "abstract is shorter and contains no detailed result table. The "
              "contribution list has four items. The gate analysis reports "
              "the fixed blend at {abl_gate_fixed} against {abl_full} for the "
              "adaptive full bridge and no longer claims that adaptivity is "
              "necessary. Figure 1 and its caption distinguish the evaluated "
              "prefix mode from optional routing strategies."),
        ("H2", "6. DOI-assigning code archive"),
        ("P", "Response: A dedicated Code Availability section now accompanies "
              "Data Availability. The corrected code, raw results, and "
              "reproduction scripts are public at "
              "https://github.com/c1326239247-coder/lab, and the archived "
              "releases are deposited in Zenodo at "
              "https://doi.org/10.5281/zenodo.23974489, which resolves to the "
              "latest version. The Code Availability section quotes that DOI."),
        ("H1", "Reviewer 1"),
        ("H2", "1. Uniform prefix burden and DynamicKV"),
        ("P", "Response: Sections 2.3, 4.8, and 5.3 now cite and discuss "
              "DynamicKV. The new matched-length and joint K/V controls test "
              "parts of the proposed explanation. Their limits are stated: "
              "the scaling intervention also changes value magnitudes, and "
              "layer-adaptive retention remains a hypothesis for future work."),
        ("H2", "2. Validation beyond TinyLlama"),
        ("P", "Response: Section 4.9 adds the Mistral-7B check requested. "
              "Section 6 limits it to one 4-bit checkpoint, one dataset, and "
              "one run per connector. The conclusions now concern our tested "
              "configurations rather than an inherent failure of all frozen "
              "read-out paths."),
        ("H1", "Reviewer 2"),
        ("H2", "1. Answer information in the training prompt summary"),
        ("P", "Response: The leakage was present and is removed. Section 3.5 "
              "defines the question-only mask and Section 5.2 records the "
              "correction. All affected bridge experiments were rerun; the "
              "updated main values are listed under Editor major comment 1."),
        ("H2", "2. Confounded prefix length and incomplete capacity sweep"),
        ("P", "Response: The caption/blind prefix propagation bug is fixed. "
              "The main lengths are 36 versus 144 tokens, while Section 4.8 "
              "separately matches total lengths. Table 6 now extends to "
              "6,556,451 parameters. Because the best capacity row exceeds "
              "the caption point estimate, we withdrew the earlier capacity "
              "conclusion and label the sweep exploratory."),
        ("H2", "3. Attention probability does not establish causation"),
        ("P", "Response: We agree and have narrowed the claim. The new "
              "controls are reported in Section 4.8, but the joint K/V scale "
              "does not separate key-driven attention from value magnitude. "
              "We no longer call it decisive causal evidence for attention "
              "crowding. Paired bootstrap intervals are reported separately "
              "from mechanistic interpretation."),
        ("H1", "Reviewer 3"),
        ("H2", "1. Abstract length"),
        ("P", "Response: The abstract is shortened and retains the research "
              "question, main finding, controls, and limitation."),
        ("H2", "2. Contribution list"),
        ("P", "Response: The list now has four items covering the mechanism, "
              "controlled evaluation, diagnostics, and reproducibility."),
        ("H2", "3. Capacity wording"),
        ("P", "Response: Table 6 brackets the baseline budget and the text "
              "reports the higher-capacity result without claiming that "
              "capacity is irrelevant."),
        ("H2", "4. Baseline prefix length"),
        ("P", "Response: The 25-token default was an implementation bug. It "
              "is corrected to the configured 36 tokens; matched total-length "
              "controls are reported separately from the 36-versus-144 main "
              "comparison."),
        ("H2", "5. Attention contribution statement"),
        ("P", "Response: The contribution and analysis now agree with the "
              "probe: the injected prefix receives high attention mass. We do "
              "not describe it as receiving lower attention than the "
              "embedding-space prefix."),
        ("H2", "6. Adaptive gate versus prompt-derived signal"),
        ("P", "Response: The fixed-blend control has a higher point estimate "
              "than the adaptive bridge ({abl_gate_fixed} versus {abl_full}). "
              "We therefore do not claim a necessary benefit from the learned "
              "coefficient. We also state that the no-gate row does not "
              "separate prompt-signal removal from parameter removal."),
        ("H2", "7. Optional routing in Figure 1"),
        ("P", "Response: The caption and Sections 3.3 and 4.1 identify prefix "
              "mode as the only evaluated configuration. Interleaved and "
              "rule-based routing variants are labelled implemented but "
              "unmeasured."),
        ("H1", "Additional reporting correction"),
        ("P", "During our audit we also clarified the metric. OK-VQA uses ten "
              "human references with the VQA-style soft score. The VQAv2 "
              "source used here provides only its majority answer and GQA "
              "provides one reference, so those two results are normalised "
              "exact match. Sections 3.5 and 6 and the Table 2 caption now "
              "state this explicitly. We also audited the split: it is by "
              "question-answer pair, not by image. Shared images across the "
              "partitions are {shared_images_vqa} in VQAv2, {shared_images_ok} "
              "in OK-VQA, and {shared_images_gqa} in GQA-spatial. The revised "
              "text therefore claims held-out-question evaluation, not "
              "unseen-image generalisation. We also audited the bibliography "
              "entry by entry against the arXiv API, Crossref and OpenAlex, "
              "and the check is now part of the repository "
              "(experiments/verify_references.py, with the findings recorded "
              "in results/reference_audit.json). It corrected five entries: "
              "P-Tuning v2 is an arXiv preprint rather than an ACL 2022 paper; "
              "the VQAv2, OK-VQA and GQA dataset papers now carry the page "
              "ranges and DOIs from their IEEE records; the multimodal survey "
              "is cited in its National Science Review journal version; and a "
              "NeurIPS page range we could not verify was removed. Resolvable "
              "DOIs were added to the remaining entries where they exist."),
    ]
    unresolved = []
    filled = []
    for kind, text in body:
        try:
            rendered = text.format(**values)
        except KeyError as exc:
            raise RuntimeError("missing response value %s" % exc) from exc
        if "{" in rendered or "}" in rendered or "@@" in rendered:
            unresolved.append(rendered[:80])
        filled.append((kind, rendered))
    if unresolved:
        raise RuntimeError("unresolved response placeholders: %s" % unresolved)
    return title, filled


def build_docx(title, body):
    """Create a clean response letter using the manuscript's visual idiom."""
    doc = docx.Document()
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, side, Inches(1.0))
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 16), ("Heading 1", 13), ("Heading 2", 11)):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name != "Title"
        style.paragraph_format.keep_with_next = True
    para = doc.add_paragraph(style="Title")
    for p_pr in (para._p.get_or_add_pPr(), doc.styles["Title"].element.pPr):
        if p_pr is not None:
            border = p_pr.find(qn("w:pBdr"))
            if border is not None:
                p_pr.remove(border)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for index, line in enumerate(title.split("\n")):
        if index:
            para.add_run().add_break()
        para.add_run(line)
    for kind, text in body:
        style = {"H1": "Heading 1", "H2": "Heading 2"}.get(kind, "Normal")
        doc.add_paragraph(text, style=style)
    path = STEM + ".docx"
    doc.save(path)
    return path


def build():
    title, body = items()
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["# " + title.replace("\n", "\n\n"), ""]
    for kind, text in body:
        prefix = {"H1": "## ", "H2": "### "}.get(kind, "")
        lines.extend([prefix + text, ""])
    md_path = STEM + ".md"
    with io.open(md_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    docx_path = build_docx(title, body)
    return md_path, docx_path


if __name__ == "__main__":
    print("written ->", build())
