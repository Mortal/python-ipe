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
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    base, ext = os.path.splitext(args.filename)
    if ext == '.tif':
        parser.error("Input and output filenames are equal")
    output_filename = base + '.tif'

    with open(args.filename) as fp:
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

    print("Writing raster of shape %s to %s" % (r.shape, output_filename))
    raster.write_generated_raster(output_filename, r)


if __name__ == "__main__":
    main()
