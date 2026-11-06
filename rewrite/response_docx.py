# -*- coding: utf-8 -*-
"""Render the reviewer response as a Word letter in the manuscript's idiom."""

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def save(title, body, path):
    """Write the response letter, as (kind, text) pairs, to a .docx path."""
    document = docx.Document()
    section = document.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    for side in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, side, Inches(1.0))
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for name, size in (("Title", 16), ("Heading 1", 13), ("Heading 2", 11)):
        style = document.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = name != "Title"
        style.paragraph_format.keep_with_next = True
    para = document.add_paragraph(style="Title")
    for p_pr in (para._p.get_or_add_pPr(),
                 document.styles["Title"].element.pPr):
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
        document.add_paragraph(text, style=style)
    document.save(path)
    return path
