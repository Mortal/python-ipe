from __future__ import division

import numpy as np
import raster
import argparse
import collections
from itertools import izip as zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-one', required=True,
                        help='First watershed raster')
    parser.add_argument('--input-two', required=True,
                        help='Second watershed raster')
    args = parser.parse_args()

    rows0 = raster.iterrows(args.input_one, pi=raster.dummy_progress)
    rows1 = raster.iterrows(args.input_two)
    active = {}
    for i, (row0, row1) in enumerate(zip(rows0, rows1)):
        tr = {}
        for j, (w0, w1) in enumerate(zip(row0, row1)):
            old = tr.setdefault(w0, w1)
            if old != w1:
                raise ValueError(
                    "Row %s column %s disagreement: %s = %s and %s" %
                    (i, j, w0, old, w1))
        for k, v0 in active.items():
            try:
                v1 = tr[k]
            except KeyError:
                print("%s %s" % (k, v0))
            else:
                if v0 != v1:
                    raise ValueError(
                        "Row %s disagreement with older row: %s = %s and %s" %
                        (i, k, v0, v1))
        active = tr
    for k, v0 in active.items():
        print('%s %s' % (k, v0))


if __name__ == "__main__":
    main()
