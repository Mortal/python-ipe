from __future__ import division

import numpy as np
import raster
import argparse
import collections
from itertools import izip as zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-elev', required=True,
                        help='Elevation model')
    parser.add_argument('--input-depths', required=True,
                        help='Water depths')
    parser.add_argument('--input-watersheds', required=True,
                        help='Watersheds')
    args = parser.parse_args()

    elev = raster.iterrows(args.input_elev, pi=raster.dummy_progress)
    depths = raster.iterrows(args.input_depths, pi=raster.dummy_progress)
    watersheds = raster.iterrows(args.input_watersheds)
    active0 = {}
    active1 = {}
    min_welev = {}
    max_welev = {}
    total_rain = 0
    for i, (z_row, d_row, w_row) in enumerate(zip(elev, depths, watersheds)):
        acc0 = collections.defaultdict(int)
        acc1 = collections.defaultdict(np.float32)
        rowzip = zip(z_row.tolist(), d_row.tolist(), w_row.tolist())
        w_nodata = raster.get_nodata_value(w_row.dtype)
        d_nodata = raster.get_nodata_value(d_row.dtype)
        for j, (z, d, w) in enumerate(rowzip):
            if w == w_nodata:
                if d != 0 and d != d_nodata:
                    print("Row %s column %s: water depth %s" %
                          (i, j, d))
            else:
                if d == d_nodata:
                    d = 0
                else:
                    welev = z + d
                    min_welev[w] = min(min_welev.get(w, welev), welev)
                    max_welev[w] = max(max_welev.get(w, welev), welev)
                acc0[w] += 1
                acc1[w] += d
        res0 = []
        res1 = []
        for k, v0 in acc0.items():
            v1 = acc1[k]
            res0.append((k, v0 + active0.pop(k, 0)))
            res1.append((k, v1 + active1.pop(k, 0)))
        for k, v0 in active0.items():
            if k != 0:
                print(('{"watershed": %s, "area": %s, "total_rain": %s, ' +
                       '"min": %s, "max": %s}') %
                      (k, v0, active1.pop(k), min_welev.pop(k),
                       max_welev.pop(k)))
            total_rain += v1
        active0 = dict(res0)
        active1 = dict(res1)
    for k, v0 in active0.items():
        v1 = active1[k]
        print('{"watershed": %s, "area": %s, "total_rain": %s}' %
              (k, v0, v1))
        total_rain += v1
    print('{"total_rain": %s}' % total_rain)


if __name__ == "__main__":
    main()
