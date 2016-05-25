from __future__ import division

import numpy as np
import raster
import argparse
import collections
from itertools import izip as zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-heights', required=True,
                        help='Water heights')
    parser.add_argument('--input-watersheds', required=True,
                        help='Watersheds')
    args = parser.parse_args()

    heights = raster.iterrows(args.input_heights)
    watersheds = raster.iterrows(args.input_watersheds)
    active0 = {}
    active1 = {}
    for h_row, w_row in zip(heights, watersheds):
        acc0 = collections.defaultdict(int)
        acc1 = collections.defaultdict(np.float32)
        for h, w in zip(h_row, w_row):
            if not raster.is_nodata(w):
                acc0[w] += 1
                acc1[w] += 0 if raster.is_nodata(h) else h
        res0 = []
        res1 = []
        for k, v0 in acc0.items():
            v1 = acc1[k]
            res0.append((k, v0 + active0.pop(k, 0)))
            res1.append((k, v1 + active1.pop(k, 0)))
        for k, v0 in active0.items():
            v1 = active1[k]
            print('{"watershed": %s, "area": %s, "total_rain": %s}' %
                  (k, v0, v1))
        active0 = dict(res0)
        active1 = dict(res1)
    for k, v0 in active0.items():
        v1 = active1[k]
        print('{"watershed": %s, "area": %s, "total_rain": %s}' %
              (k, v0, v1))


if __name__ == "__main__":
    main()
