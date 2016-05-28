"""
Convert ASCII grid to GeoTIFF.

Each line in the input file becomes a corresponding row,
and the number of columns in the output raster is taken to be
the length of the longest line.

Lines must consist of digits 0-9 and spaces.
Digits are converted to integer heights.

If you want floating-point heights, you can use TerraSTREAM's Utility-convert,
parameters --output bla.txt --output-type gdal --output-gdal-driver AAIGrid
"""
import os
import sys
import argparse
import textwrap

import numpy as np


PY2 = sys.hexversion < 0x03000000


def main():
    try:
        import raster
    except ImportError:
        if not PY2:
            print("This is a Python 2 script. Rerunning with Python 2.")
            os.execlp('python2', 'python2', *sys.argv)
        raise
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+')
    args = parser.parse_args()
    for f in [convert(raster, f) for f in args.filename]:
        f()


def convert(raster, filename):
    base, ext = os.path.splitext(filename)
    if ext == '.tif':
        parser.error("Input and output filenames are equal")
    output_filename = base + '.tif'

    with open(filename) as fp:
        terrain = fp.read().strip('\n').splitlines()

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

    def defer():
        print("Writing raster of shape %s to %s" % (r.shape, output_filename))
        raster.write_generated_raster(output_filename, r)

    return defer


if __name__ == "__main__":
    main()
