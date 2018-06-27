import sys
import argparse

from xml.etree import ElementTree

import ipe.shape


def unmatrix(root, matrix=None):
    if matrix is None:
        matrix = ipe.shape.MatrixTransform.identity()
    if 'matrix' in root.attrib:
        m = ipe.shape.load_matrix(root.attrib['matrix'])
        del root.attrib['matrix']
        matrix = m.and_then(matrix)
    if root.tag == 'path':
        root.text = ipe.shape.apply_matrix_to_shape(root.text, matrix)
    elif root.tag == 'group':
        for child in root:
            unmatrix(child, matrix)
    elif root.tag in ('text', 'use'):
        x, y = root.attrib['pos'].split()
        x, y = matrix.transform(float(x), float(y))
        root.attrib['pos'] = '%s %s' % (x, y)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    d = sys.stdin.read()
    root = ElementTree.fromstring(d)
    unmatrix(root)
    print(ElementTree.tostring(root).decode())


if __name__ == "__main__":
    main()
