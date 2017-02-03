#!/usr/bin/env python3
'''
Use raster.py write_generated_raster to make a small raster for testing.
'''

import textwrap

import numpy as np

import raster


def main():
    terrain = textwrap.dedent("""
    ### ### #####
    #.# #.###...##
    #.# ##...###.#
    #.#  ##### #.#
    #.#        #.#
    #.##########.#
    ##..........##
     ############
    """).strip('\n').splitlines()

    nrows = len(terrain)
    ncols = max(len(row) for row in terrain)
    r = np.zeros((nrows, ncols), dtype=np.float32)
    r = raster.nodata_like(r)
    nodata = r[0, 0]
    dots = []
    for i, row in enumerate(terrain):
        for j, cell in enumerate(row):
            if cell == '#':
                r[i, j] = 40
            elif cell == '.':
                dots.append((i, j))
    dfs = [(1, 5)]
    h = 1
    dd = [(i - 1, j - 1) for i in range(3) for j in range(3) if not (i == j == 1)]
    while dfs:
        i, j = dfs.pop()
        r[i, j] = h
        h += 1
        for di, dj in dd:
            if r[i+di, j+dj] == nodata:
                dfs.append((i+di, j+dj))
    raster.write_generated_raster('raster7.tif', r)


if __name__ == "__main__":
    main()
