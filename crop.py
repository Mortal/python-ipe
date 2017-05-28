#!/usr/bin/env python3
import re
import argparse
import itertools
import collections
from raster import iterrows, raster_sink, peek_dtype


Spec = collections.namedtuple('Spec', 'width height left top')


def parse_spec(s):
    mo = re.parse(r'^(\d+)x(\d+)\+(\d+)\+(\d+)$', s)
    if not mo:
        raise ValueError('format: WIDTHxHEIGHT+LEFT+TOP')
    return Spec(*map(int, mo.groups()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--spec', required=True,
                        help='WIDTHxHEIGHT+LEFT+TOP', type=parse_spec)
    args = parser.parse_args()

    meta, rows = iterrows(args.input, meta=True)
    dtype, rows = peek_dtype(rows)
    rows = itertools.islice(rows, args.spec.top,
                            args.spec.top + args.spec.height)
    i = args.spec.left
    j = args.spec.left + args.spec.width
    rows = (row[i:j] for row in rows)
    raster_sink(args.output, rows, dtype, meta)


if __name__ == '__main__':
    main()
