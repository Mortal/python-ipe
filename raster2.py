#!/usr/bin/env python3
'''
Use raster.py write_generated_raster to make a small raster for testing
nodata-all-high.
'''

import textwrap

import numpy as np

import raster


def main():
    terrain = textwrap.dedent("""
    00 00 0
    0111110
    0122210
     12 21
    0122210
    0111110
    00 00 0
    """).strip('\n').splitlines()

    nrows = len(terrain)
    ncols = max(len(row) for row in terrain)
    r = np.zeros((nrows, ncols), dtype=np.float32)
    r = raster.nodata_like(r)
    nodata = r[0, 0]
    dots = []
    for i, row in enumerate(terrain):
        for j, cell in enumerate(row):
            if cell != ' ':
                r[i, j] = float(cell)
    raster.write_generated_raster('raster2.tif', r)


if __name__ == "__main__":
    main()
