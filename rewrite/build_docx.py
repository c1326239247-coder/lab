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
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import content
import numbers as facts_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = ("C:/Users/luyang/Downloads/"
       "Cache-Bridging Lossless Cross-Modal Transfer via 8.21.docx")
DST = ("C:/Users/luyang/Downloads/"
       "Cache-Bridging Hierarchical Visual KV Injection (revised).docx")
FIG_DIR = os.path.join(ROOT, "paper_figures")


def fill(text, values):
    try:
        return text.format(**values)
    except (KeyError, IndexError) as exc:
        raise SystemExit("unresolved placeholder in: %r (%s)" % (text[:90], exc))


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

    def caption(self, text):
        """A figure/table caption: bold label, plain remainder, centred."""
        para = self.para("", align=WD_ALIGN_PARAGRAPH.CENTER)
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
        para.add_run().add_picture(path, width=Inches(width_inches))
        return para

    def table(self, rows, style, widths=None):
        table = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        try:
            table.style = self.doc.styles[style]
        except KeyError:
            pass
        table.autofit = True
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cell = table.cell(r, c)
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
    """The five data tables, all read straight from results/*.json."""
    abl = runs["abl"]
    back = runs["back"]
    n = back["clip_num_layers"]
    taps = [("basic", max(1, n // 6), "early texture and edge statistics"),
            ("semantic_object", n // 2, "mid-level object parts"),
            ("spatial_relation", int(n * 0.75), "layout and relational features"),
            ("attribute", n - 2, "late, globally contextualised attributes")]

    levels = [["Cache level", "CLIP layer", "What it is expected to carry"]]
    for name, idx, meaning in taps:
        levels.append([name, idx, meaning])

    main = [["Connector", "Trainable params", "VQAv2", "OK-VQA", "GQA-spatial"]]
    main.append(["Caption-style projection (baseline)",
                 "{:,}".format(runs["vqa_c"]["connector_params"]),
                 "%.2f" % runs["vqa_c"]["accuracy"],
                 "%.2f" % runs["ok_c"]["accuracy"],
                 "%.2f" % runs["gqa_c"]["accuracy"]])
    main.append(["Cache-bridging (this work)",
                 "{:,}".format(runs["vqa_b"]["connector_params"]),
                 "%.2f" % runs["vqa_b"]["accuracy"],
                 "%.2f" % runs["ok_b"]["accuracy"],
                 "%.2f" % runs["gqa_b"]["accuracy"]])
    main.append(["Cache-bridging, larger bridge (Section 4.6)",
                 "{:,}".format(runs["gqa_big"]["connector_params"]),
                 "\u2014", "\u2014",
                 "%.2f" % runs["gqa_big"]["accuracy"]])

    def cell(run, key):
        value = run.get(key)
        return "\u2014" if value is None else "%.2f" % value

    control = [["Condition", "VQAv2", "OK-VQA", "GQA-spatial"]]
    control.append(["Cache-bridging, correct image"]
                   + [cell(runs[k], "accuracy") for k in ["vqa_b", "ok_b", "gqa_b"]])
    control.append(["Cache-bridging, mismatched image"]
                   + [cell(runs[k], "accuracy_shuffled_images")
                      for k in ["vqa_b", "ok_b", "gqa_b"]])
    control.append(["Cache-bridging, visual cache disabled"]
                   + [cell(runs[k], "accuracy_no_visual")
                      for k in ["vqa_b", "ok_b", "gqa_b"]])
    control.append(["Baseline, correct image"]
                   + [cell(runs[k], "accuracy") for k in ["vqa_c", "ok_c", "gqa_c"]])
    control.append(["Baseline, mismatched image"]
                   + [cell(runs[k], "accuracy_shuffled_images")
                      for k in ["vqa_c", "ok_c", "gqa_c"]])

    order = ["text only (no visual pathway)", "caption-style projection",
             "cache-bridging, full cache", "cache-bridging, 50% top-K",
             "cache-bridging, 25% top-K"]
    labels = {"text only (no visual pathway)": "Text only (no visual pathway)",
              "caption-style projection": "Caption-style projection",
              "cache-bridging, full cache": "Cache-bridging, full cache",
              "cache-bridging, 50% top-K": "Cache-bridging, Top-K keeping 50%",
              "cache-bridging, 25% top-K": "Cache-bridging, Top-K keeping 25%"}
    by_system = dict((v["system"], v) for v in runs["cost"]["variants"])
    cost = [["System", "Latency (ms)", "Median (ms)", "Peak memory (GB)"]]
    for key in order:
        v = by_system[key]
        cost.append([labels[key], "%.1f" % v["latency_ms_mean"],
                     "%.1f" % v["latency_ms_median"], "%.2f" % v["peak_mem_gb"]])

    base = abl["full"]["accuracy"]
    rows = [("full", "Full bridge, all four levels"),
            ("no_geometry", "without the 2D geometry bias"),
            ("no_gate", "without the gate (parameters removed with it)"),
            ("gate_fixed", "gate fixed to an even blend (matched parameters)"),
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
    ablation = [["Variant", "GQA-spatial", "\u0394", "Trainable params"]]
    for tag, label in rows:
        run = abl[tag]
        delta = run["accuracy"] - base
        ablation.append([label, "%.2f" % run["accuracy"],
                         "%+.2f" % delta if tag != "full" else "\u2014",
                         "{:,}".format(run["connector_params"])])

    return {"levels": levels, "main": main, "control": control,
            "cost": cost, "ablation": ablation}


TABLE_CAPTIONS = {
    "levels": ("Table 1. The four cache levels, the CLIP-ViT-L/14 encoder layer "
               "each is tapped from, and the kind of information the layer-wise "
               "probing literature associates with that depth."),
    "main": ("Table 2. VQA accuracy (%) on {eval_size} held-out questions per "
             "dataset, after {steps} steps under an identical training protocol "
             "with both backbones frozen. Higher is better."),
    "control": ("Table 3. Control conditions (%). \u201cMismatched image\u201d scores the "
                "same trained weights with every question paired with another "
                "question\u2019s image; \u201cvisual cache disabled\u201d zeroes the level mask, "
                "which is only possible for the bridge."),
    "cost": ("Table 4. Single-sample inference cost, {cost_repeats} questions, "
             "{cost_new_tokens} generated tokens, measured on one {gpu}."),
    "ablation": ("Table 5. Ablations on GQA-spatial at {abl_steps} steps. Each "
                 "variant changes one component; \u0394 is relative to the full "
                 "bridge."),
}

TABLE_STYLES = {"levels": "Light Grid", "main": "Grid Table 6 Colorful",
                "control": "Grid Table 6 Colorful",
                "cost": "Grid Table 6 Colorful",
                "ablation": "Grid Table 6 Colorful"}


# Paragraph indices in the ORIGINAL document (see the map printed by
# rewrite/inspect_source.py).  Everything strictly between the abstract
# and the reference heading is replaced; the four figure paragraphs are
# lifted out first and put back where content.py asks for them.
IDX_TITLE = (0, 1)
IDX_ABSTRACT = 16
IDX_KEYWORDS = 17
IDX_BODY_FIRST = 18          # "1 Introduction"
IDX_REFS_HEADING = 108
IDX_FIGURES = (21, 44, 50, 61)
IDX_DECL_FIRST = 138         # blank line before "Funding"
IDX_DATA_LABEL = 145
IDX_DATA_TEXT = 146
IDX_APPENDIX_FIRST = 147


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
    doc = docx.Document(SRC)
    paras = doc.paragraphs

    # Title and abstract.
    set_text(paras[IDX_TITLE[0]], content.TITLE[0])
    set_text(paras[IDX_TITLE[1]], content.TITLE[1])
    set_text(paras[IDX_ABSTRACT], fill(content.ABSTRACT, values))
    set_text(paras[IDX_KEYWORDS], content.KEYWORDS)

    # Detach the four figure images so they can be re-placed by content.py.
    figures = []
    for idx in IDX_FIGURES:
        element = paras[idx]._p
        element.getparent().remove(element)
        figures.append(element)

    # Drop the old body, the old declarations, and the old appendix.
    removed = set(IDX_FIGURES)
    for idx in list(range(IDX_BODY_FIRST, IDX_REFS_HEADING)) + \
            list(range(IDX_DECL_FIRST, len(paras))):
        if idx in removed:
            continue
        element = paras[idx]._p
        if element.getparent() is not None:
            element.getparent().remove(element)
    for table in list(doc.tables):
        table._tbl.getparent().remove(table._tbl)

    # Rebuild the body after the keyword line.
    builder = Builder(doc, paras[IDX_KEYWORDS])
    for kind, payload in content.BODY:
        if kind == "H1":
            builder.para(payload, style="Heading 1")
        elif kind == "H2":
            builder.para(payload, style="Heading 2")
        elif kind == "P":
            builder.para(fill(payload, values))
        elif kind == "B":
            builder.para(fill(payload, values), style="List Bullet")
        elif kind == "FIG":
            builder.move(figures[payload])
        elif kind == "FIGNEW":
            builder.picture(os.path.join(FIG_DIR, payload))
        elif kind == "CAP":
            builder.caption(fill(payload, values))
        elif kind == "TBL":
            builder.caption(fill(TABLE_CAPTIONS[payload], values))
            builder.table(tables[payload], TABLE_STYLES[payload])
            builder.para("")
        else:
            raise SystemExit("unknown item kind %r" % kind)

    # Declarations and appendix, after the reference list.
    tail = doc.paragraphs[-1]
    builder = Builder(doc, tail)
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
            ("Data Availability", content.DATA_AVAILABILITY)]:
        builder.para(label, bold=True, keep_with_next=True)
        builder.para(text)
    builder.para("")
    for kind, payload in content.APPENDIX:
        if kind == "H1":
            builder.para(payload, style="Heading 1")
        elif kind == "P":
            builder.para(fill(payload, values))
        elif kind == "B":
            builder.para(fill(payload, values), style="List Bullet")
        else:
            raise SystemExit("unknown appendix kind %r" % kind)

    doc.save(DST)
    return DST, values


if __name__ == "__main__":
    path, values = build()
    print("written ->", path)
