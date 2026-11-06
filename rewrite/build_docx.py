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
from manuscript_citations import Citations, fill
import manuscript_tables
import numbers as facts_mod
import refs as refs_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "rewrite", "template.docx")
DIST = os.path.join(ROOT, "dist")
DST = os.path.join(DIST, "Cache-Bridging_revised_2026-10-30.docx")
FIG_DIR = os.path.join(ROOT, "paper_figures")


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
        if style not in [s.name for s in self.doc.styles]:
            raise SystemExit("table style %r is missing from the template"
                             % style)
        table.style = self.doc.styles[style]
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
    if not os.path.exists(SRC):
        raise SystemExit(
            "layout template missing: the Word file this manuscript is written "
            "into is not shipped in the public repository. Place the submission "
            ".docx at %s, or read the numbers from results/ and the prose from "
            "rewrite/content.py instead." % SRC)
    values, runs = facts_mod.facts()
    tables = manuscript_tables.build_tables(values, runs)
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
            builder.caption(fill(manuscript_tables.TABLE_CAPTIONS[payload],
                                 values, cites),
                            keep_with_next=True)
            builder.table(tables[payload],
                          manuscript_tables.TABLE_STYLES[payload])
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
