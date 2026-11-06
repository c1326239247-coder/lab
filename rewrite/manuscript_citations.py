# -*- coding: utf-8 -*-
"""Citation numbering for the manuscript.

The body cites @@CITE:key@@ markers; this module assigns reference numbers
in order of first appearance and aborts the build on an unknown key, an
orphaned entry or an unresolved marker."""

import re


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
