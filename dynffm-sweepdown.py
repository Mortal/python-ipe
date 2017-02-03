#!/usr/bin/env python3
'''
Program to compare the flash flood map output by Jungwoo's code
with the intermediate "flooded subtrees" output of Mathias's rain-event code.

Uses iterrows of raster.py to read rasters.
'''

from __future__ import division

import os
import re
import json
import raster
import argparse
# import collections
# import numpy as np
from itertools import izip as zip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dynffm', required=True,
                        help='Depths from dynffm (GeoTIFF)')
    parser.add_argument('--input-dfs-watersheds', required=True,
                        help='Watersheds (GeoTIFF)')
    parser.add_argument('--input-flooded-subtrees', required=True,
                        help='Flooded subtree list (text)')
    args = parser.parse_args()

    base, ext = os.path.splitext(args.input_dynffm)
    cache_file = base + '-sum-by-watershed.json'
    if os.path.exists(cache_file):
        with open(cache_file) as fp:
            volumes = dict(json.load(fp))
    else:
        depths = raster.iterrows(args.input_dynffm)
        watersheds = raster.iterrows(
            args.input_dfs_watersheds, pi=raster.dummy_progress)
        volumes = {}

        try:
            for i, (d_row, w_row) in enumerate(zip(depths, watersheds)):
                nodata = raster.get_nodata_value(d_row.dtype)
                for j, (d, w) in enumerate(zip(d_row.tolist(), w_row.tolist())):
                    if d == nodata:
                        d = 0
                    try:
                        volumes[w] += d
                    except KeyError:
                        volumes[w] = d
        except KeyboardInterrupt:
            pass
        else:
            with open(cache_file, 'w') as fp:
                json.dump(list(volumes.items()), fp)

    pattern = (r'(\d+):(\d+) (?:full|partially_full) V=([-+e.\d]+) ' +
               r'basinVolume=[-+e.\d]+ z=[-+e.\d]+')
    with open(args.input_flooded_subtrees) as fp:
        for line in fp:
            if line.startswith('Total rain is '):
                continue
            mo = re.match(pattern, line)
            if mo is None:
                raise ValueError(line)
            a = int(mo.group(1))
            b = int(mo.group(2))
            sweepdown_volume = mo.group(3)
            values = [volumes.get(i, 0) for i in range(a, b+1)]
            if not any(values):
                continue
            dynffm_volume = sum(values)
            if ('%g' % dynffm_volume) == sweepdown_volume:
                diff = 0
            else:
                diff = (float(dynffm_volume) - float(sweepdown_volume)) / float(dynffm_volume)
            print("%s %s %s" % (diff, dynffm_volume, line.rstrip('\n')))


if __name__ == "__main__":
    main()
