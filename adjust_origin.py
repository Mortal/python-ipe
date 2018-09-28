"""
Given an Ipe selection of a path and a text, where both have
transformations="translations", move the origin of the path to the text.
This essentially groups the two objects together
when the entire drawing is scaled.

Only tested for polylines, not for arcs/curves.
"""

import re
import sys
from xml.etree import ElementTree as ET


def get_origin(o):
    x, y = map(float, (o.get("pos") or "0 0").split())
    a, b, c, d, e, f = map(float, (o.get("matrix") or "1 0 0 1 0 0").split())
    assert o.get("transformations") == "translations"
    return a * x + c * y + e, b * x + d * y + f


def main():
    s = sys.stdin.read()
    tree = ET.fromstring(s)
    assert tree.tag == "ipeselection"
    path, point = sorted(tree, key=lambda n: n.tag)
    assert path.tag == "path"
    assert point.tag in ("text", "use")
    px, py = get_origin(path)
    tx, ty = get_origin(point)
    path.set("matrix", "1 0 0 1 %g %g" % (tx, ty))

    def translate(mo):
        return "%g %g" % (float(mo.group(1)) + px - tx, float(mo.group(2)) + py - ty)

    path.text = re.sub(r"(-?[0-9.]+) (-?[0-9.]+)", translate, path.text)
    print(ET.tostring(tree, encoding="unicode", method="xml"))


if __name__ == "__main__":
    main()
