# -*- coding: utf-8 -*-
"""Label the evaluated and optional routes on the retained Figure 1."""

import io
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != SCRIPT_DIR]

import docx
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "rewrite", "template.docx")
DST = os.path.join(ROOT, "paper_figures", "fig1_prefix_evaluated.png")


def load_font(size, bold=False):
    """Use Arial where available and a readable open fallback elsewhere."""
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else (
        "arial.ttf", "DejaVuSans.ttf")
    candidates = []
    for name in names:
        candidates.extend([
            os.path.join("C:/Windows/Fonts", name),
            os.path.join("/usr/share/fonts/truetype/dejavu", name),
            os.path.join("/System/Library/Fonts/Supplemental", name),
            name,
        ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    document = docx.Document(SRC)
    shape = list(document.inline_shapes)[0]
    rel_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    image = Image.open(io.BytesIO(
        document.part.related_parts[rel_id].blob)).convert("RGB")
    banner = 190
    output = Image.new("RGB", (image.width, image.height + banner), "white")
    output.paste(image, (0, banner))
    draw = ImageDraw.Draw(output)
    regular = load_font(48)
    bold = load_font(60, bold=True)
    draw.text((32, 24), "EVALUATED: full-level prefix injection",
              fill=(20, 70, 110), font=bold)
    draw.text((32, 104),
              "OPTIONAL, IMPLEMENTED BUT UNMEASURED: selective and progressive routing",
              fill=(70, 70, 70), font=regular)
    draw.line((0, banner - 1, image.width, banner - 1),
              fill=(190, 190, 190), width=2)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    output.save(DST)
    print("written ->", DST)


if __name__ == "__main__":
    main()
