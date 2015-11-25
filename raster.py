from __future__ import division

import os
import sys
import gdal


gdal.UseExceptions()


def show_progress(name=""):
    def pi(i, n):
        sys.stdout.write("%3d%% %s %12d/%d\r" % (i * 100 / n, name, i, n))
        sys.stdout.flush()
        if i == n:
            print('')

    return pi


def iterrows(filename, pi=None):
    if pi is None:
        pi = show_progress(os.path.basename(filename))
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    row = band.ReadAsArray(0, 0, win_ysize=1)
    nrows = band.ReadAsArray(0, 0, win_xsize=0).shape[0]
    progress = -1
    for i in range(nrows):
        band.ReadAsArray(0, i, win_ysize=1, buf_obj=row)
        yield row.ravel()
        p = (i + 1) * 1000 // nrows
        if p > progress or i + 1 == nrows:
            progress = p
            pi(i + 1, nrows)
