# -*- coding: utf-8 -*-
"""Rebuild the manuscript from rewrite/content.py and results/*.json.

The original .docx is used as the template so that the page setup, the
styles, the author block, the reference list, and the four existing
figures are preserved; only the body between the abstract and the
reference list, and the declarations after it, are rewritten.
"""

import copy
import io
import os
import re
import sys

import docx
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import content
import numbers as facts_mod
import refs as refs_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "rewrite", "template.docx")
DIST = os.path.join(ROOT, "dist")
DST = os.path.join(DIST, "Cache-Bridging_revised_2026-10-30.docx")
FIG_DIR = os.path.join(ROOT, "paper_figures")


CITE = re.compile(r"@@CITE:([a-zA-Z0-9_,]+)@@")


class Citations(object):
    """Assigns reference numbers in order of first appearance.

    The body cites @@CITE:key@@ markers instead of literal numbers, so a
    reference cannot end up attached to the wrong entry, an entry cannot be
    silently orphaned, and the numbering always matches the reading order.
    """

    def __init__(self, table):
        self.table = table
        self.order = []

    def number(self, key):
        if key not in self.table:
            raise SystemExit("citation to unknown reference %r" % key)
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def substitute(self, text):
        def repl(match):
            keys = [k for k in match.group(1).split(",") if k]
            numbers = sorted(self.number(k) for k in keys)
            return "[" + ", ".join(str(n) for n in numbers) + "]"
        return CITE.sub(repl, text)

    def unused(self):
        return [key for key in self.table if key not in self.order]


def fill(text, values, cites=None):
    try:
        text = text.format(**values)
    except (KeyError, IndexError) as exc:
        raise SystemExit("unresolved placeholder in: %r (%s)" % (text[:90], exc))
    if cites is not None:
        text = cites.substitute(text)
    if "@@" in text:
        raise SystemExit("unresolved citation marker in: %r" % text[:90])
    return text


class Builder(object):
    """Appends paragraphs and tables in document order after an anchor."""

    def __init__(self, doc, anchor):
        self.doc = doc
        self.cursor = anchor._p

    def _insert(self, element):
        self.cursor.addnext(element)
        self.cursor = element
        return element

    def para(self, text, style=None, align=None, italic=False, bold=False,
             keep_with_next=False):
        template = self.doc.add_paragraph()
        element = self._insert(template._p)
        para = docx.text.paragraph.Paragraph(element, self.doc._body)
        if style:
            para.style = self.doc.styles[style]
        if align is not None:
            para.alignment = align
        if text:
            run = para.add_run(text)
            run.italic = italic
            run.bold = bold
        if keep_with_next:
            para.paragraph_format.keep_with_next = True
        return para

    def caption(self, text, keep_with_next=False):
        """A figure/table caption: bold label, plain remainder, centred."""
        para = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER)
        para.paragraph_format.keep_together = True
        para.paragraph_format.keep_with_next = keep_with_next
        match = re.match(r"^((?:Figure|Table)\s+\d+\.)\s*(.*)$", text, re.S)
        if match:
            para.add_run(match.group(1)).bold = True
            para.add_run(" " + match.group(2))
        else:
            para.add_run(text)
        return para

    def move(self, element):
        """Move an existing element (kept figure) to the cursor."""
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
        return self._insert(element)

    def picture(self, path, width_inches=5.6):
        para = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER)
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(path, width=Inches(width_inches))
        return para

    def picture_blob(self, blob, width_inches=5.6):
        para = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER)
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(io.BytesIO(blob),
                                   width=Inches(width_inches))
        return para

    def table(self, rows, style, widths=None):
        table = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        try:
            table.style = self.doc.styles[style]
        except KeyError:
            pass
        table.autofit = True
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r, row in enumerate(rows):
            tr_pr = table.rows[r]._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            if r == 0:
                header = OxmlElement("w:tblHeader")
                header.set(qn("w:val"), "true")
                tr_pr.append(header)
            for c, value in enumerate(row):
                cell = table.cell(r, c)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text = ""
                para = cell.paragraphs[0]
                run = para.add_run(str(value))
                if r == 0:
                    run.bold = True
                if c > 0:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._insert(table._tbl)
        return table


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
        caption_run = facts_mod.load("run_prefix%s_caption.json" % tag)
        bridge_run = facts_mod.load("run_prefix%s_bridge.json" % tag)
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


def find_index(paras, prefix):
    for idx, para in enumerate(paras):
        if para.text.strip().startswith(prefix):
            return idx
    raise SystemExit("could not locate paragraph starting with %r" % prefix)


def set_text(paragraph, text):
    """Replace a paragraph's text while keeping its first run's format."""
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def build():
    values, runs = facts_mod.facts()
    tables = build_tables(values, runs)
    cites = Citations(refs_mod.BY_KEY)
    doc = docx.Document(SRC)
    paras = doc.paragraphs

    title_idx = find_index(paras, "Cache-Bridging")
    abstract_idx = find_index(paras, "Standard vision-language pipelines")
    keywords_idx = find_index(paras, "Keywords: Cache-Bridging")
    body_first = find_index(paras, "1  Introduction")
    refs_heading = find_index(paras, "References")

    # Title and abstract.
    set_text(paras[title_idx], " ".join(content.TITLE))
    paras[title_idx].style = doc.styles["Title"]
    paras[title_idx].paragraph_format.line_spacing = 1.15
    for p_pr in (paras[title_idx]._p.get_or_add_pPr(),
                 doc.styles["Title"].element.pPr):
        if p_pr is not None:
            border = p_pr.find(qn("w:pBdr"))
            if border is not None:
                p_pr.remove(border)
    for run in paras[title_idx].runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
        run.font.size = docx.shared.Pt(18)
    for style_name in ("Title", "Heading 1", "Heading 2"):
        doc.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
    set_text(paras[abstract_idx], fill(content.ABSTRACT, values, cites))
    paras[abstract_idx].paragraph_format.keep_together = True
    abstract_heading_idx = find_index(paras, "Abstract")
    paras[abstract_heading_idx].paragraph_format.keep_with_next = True
    set_text(paras[keywords_idx], content.KEYWORDS)

    # Capture the four template figures before removing the old body.
    figures = []
    for shape in list(doc.inline_shapes)[:4]:
        rel_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        figures.append(doc.part.related_parts[rel_id].blob)
    if len(figures) != 4:
        raise SystemExit("expected four template figures")

    # Drop the old body, the old reference entries, the old declarations,
    # and the old appendix.  Only the "References" heading itself is kept.
    for idx in list(range(body_first, refs_heading)) + \
            list(range(refs_heading + 1, len(paras))):
        element = paras[idx]._p
        if element.getparent() is not None:
            element.getparent().remove(element)
    for table in list(doc.tables):
        table._tbl.getparent().remove(table._tbl)

    # Rebuild the body after the keyword line.
    builder = Builder(doc, paras[keywords_idx])
    for kind, payload in content.BODY:
        if kind == "H1":
            builder.para(payload, style="Heading 1")
        elif kind == "H2":
            builder.para(payload, style="Heading 2")
        elif kind == "P":
            builder.para(fill(payload, values, cites))
        elif kind == "B":
            builder.para(fill(payload, values, cites), style="List Bullet")
        elif kind == "FIG":
            builder.picture_blob(figures[payload])
        elif kind == "FIGNEW":
            builder.picture(os.path.join(FIG_DIR, payload))
        elif kind == "CAP":
            builder.caption(fill(payload, values, cites))
        elif kind == "TBL":
            builder.caption(fill(TABLE_CAPTIONS[payload], values, cites),
                            keep_with_next=True)
            builder.table(tables[payload], TABLE_STYLES[payload])
            builder.para("")
        else:
            raise SystemExit("unknown item kind %r" % kind)

    # Reference list, numbered in order of first citation.
    builder = Builder(doc, paras[refs_heading])
    for number, key in enumerate(cites.order, start=1):
        builder.para("[%d] %s" % (number, refs_mod.BY_KEY[key]))
    orphans = cites.unused()
    if orphans:
        raise SystemExit("references present but never cited: %s"
                         % ", ".join(orphans))

    # Declarations and appendix, after the reference list.
    builder.para("")
    for label, text in [
            ("Funding", "This research received no specific grant from any funding "
                        "agency in the public, commercial, or not-for-profit "
                        "sectors."),
            ("Ethical Approval", "This article does not contain any studies with "
                                 "human participants or animals performed by any of "
                                 "the authors."),
            ("Informed Consent", "This article does not contain any studies with "
                                 "human participants performed by any of the "
                                 "authors."),
            ("Competing Interests", content.COMPETING_INTERESTS),
            ("Code Availability", content.CODE_AVAILABILITY),
            ("Data Availability", content.DATA_AVAILABILITY)]:
        builder.para(label, bold=True, keep_with_next=True)
        builder.para(text)
    builder.para("")
    for kind, payload in content.APPENDIX:
        if kind == "H1":
            builder.para(payload, style="Heading 1")
        elif kind == "P":
            builder.para(fill(payload, values, cites))
        elif kind == "B":
            builder.para(fill(payload, values, cites), style="List Bullet")
        else:
            raise SystemExit("unknown appendix kind %r" % kind)

    os.makedirs(DIST, exist_ok=True)
    doc.save(DST)
    return DST, values


if __name__ == "__main__":
    path, values = build()
    print("written ->", path)
