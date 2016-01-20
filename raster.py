from __future__ import division

import os
import sys
import gdal
import time

from itertools import izip as zip
from itertools import chain

import numpy as np


gdal.UseExceptions()


def show_progress(name=""):
    t = [0, 0, 0]
    def pi(i, n):
        if t[0] == 0:
            t[0] = time.time()
        elif i % 10 == 0:
            t[1] = time.time()
            t[2] = i / (t[1] - t[0])
        sys.stdout.write("%3d%% %s %12d/%d %g\r" %
                         (i * 100 / n, name, i, n, t[2]))
        sys.stdout.flush()
        if i == n:
            print('')

    return pi


def dummy_progress(i, n):
    pass


def iterrows(filename, pi=None, meta=False, buffer_rows=1):
    if pi is None:
        pi = show_progress(os.path.basename(filename))
    ds = gdal.Open(filename)

    def it():
        # xsize = ds.RasterXSize
        nrows = ds.RasterYSize
        band = ds.GetRasterBand(1)
        bufrows = min(buffer_rows, nrows)
        row = band.ReadAsArray(0, 0, win_ysize=bufrows)
        if band.GetNoDataValue() != get_nodata_value(row.dtype):
            print("WARNING: Incorrect nodata value: %s != %s" %
                  (band.GetNoDataValue(), get_nodata_value(row.dtype)))
        progress = -1
        for i in range(0, nrows, bufrows):
            j = min(nrows, i + bufrows)
            for k in range(j - i):
                band.ReadAsArray(0, i + k, win_ysize=1, buf_obj=row[k:k+1])
                yield row[k]
                p = (i + 1) * 1000 // nrows
                if p > progress or i + 1 == nrows:
                    progress = p
                    pi(i + 1, nrows)

    if meta:
        return ds, it()
    else:
        return it()


def raster_sink(filename, iterable, dtype, meta):
    out_driver = gdal.GetDriverByName('GTiff')
    gdal_dtype = gdal.GetDataTypeByName(dtype.__name__)

    xsize = meta.RasterXSize
    ysize = meta.RasterYSize
    nbands = 1

    assert xsize > 0
    assert ysize > 0

    ds = out_driver.Create(filename, xsize, ysize, nbands, gdal_dtype)
    ds.SetGeoTransform(meta.GetGeoTransform())
    ds.SetProjection(meta.GetProjection())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(get_nodata_value(dtype))
    for i, row in enumerate(iterable):
        band.WriteArray(row.reshape(1, xsize), 0, i)


def load(filename, pi=None):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(0, 0)


def get_nodata_value(dtype):
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).min


def nodata_like(row):
    row = np.asarray(row)
    return np.zeros_like(row) + get_nodata_value(row.dtype)


def window_single(iterable):
    try:
        row = next(iterable)
    except StopIteration:
        return
    a = nodata_like(row)
    b = nodata_like(row)
    c = nodata_like(row)
    c[:] = row
    more = True
    while more:
        a, b, c = b, c, a
        try:
            c[:] = next(iterable)
        except StopIteration:
            c[:] = nodata_like(b)
            more = False
        yield a, b, c


def window(*args):
    if len(args) == 1:
        return window_single(args[0])
    return zip(*[window_single(i) for i in args])


def add_nodata_row(iterable):
    try:
        row = next(iterable)
    except StopIteration:
        return []
    nodata_row = nodata_like(row)
    return chain([row], iterable, [nodata_row])
