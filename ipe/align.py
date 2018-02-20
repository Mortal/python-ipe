import sys
import math
import argparse

from xml.etree import ElementTree

import ipe.shape


def find_box(root, width, height):
    if root.tag == 'group':
        for child in root:
            q = find_box(child, width, height)
            if q is not None:
                return q
    if root.tag == 'path':
        shape = ipe.shape.load_shape(root.text)
        assert shape.is_polygon()
        edges = shape.get_edges()
        xs = set(a.real for u, v in edges for a in (u, v))
        ys = set(a.imag for u, v in edges for a in (u, v))
        x1, x2 = xs
        y1, y2 = ys
        return x1, (x2 - x1) / width, y1, (y2 - y1) / height


def recursive_align(root, matrix, q):
    x1, dx, y1, dy = q
    if 'matrix' in root.attrib:
        m = ipe.shape.load_matrix(root.attrib['matrix'])
        del root.attrib['matrix']
        matrix = m.and_then(matrix)
    if root.tag == 'text':
        x, y = root.attrib['pos'].split()
        x, y = matrix.transform(float(x), float(y))
        i = math.floor((x - x1) / dx) + 0.5
        j = math.floor((y - y1) / dy) + 0.5
        root.attrib['pos'] = '%s %s' % (x1 + dx * i, y1 + dy * j)
    elif root.tag == 'group':
        for child in root:
            recursive_align(child, matrix, q)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    args = parser.parse_args()
    d = sys.stdin.read()
    root = ElementTree.fromstring(d)
    q = find_box(root, args.width, args.height)
    recursive_align(root, ipe.shape.MatrixTransform.identity(), q)
    print(ElementTree.tostring(root).decode())


if __name__ == "__main__":
    main()
