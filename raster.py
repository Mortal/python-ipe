from __future__ import division

import os
import sys
import gdal

from itertools import izip as zip

import numpy as np


gdal.UseExceptions()


def show_progress(name=""):
    def pi(i, n):
        sys.stdout.write("%3d%% %s %12d/%d\r" % (i * 100 / n, name, i, n))
        sys.stdout.flush()
        if i == n:
            print('')

    return pi


def dummy_progress(i, n):
    pass


def iterrows(filename, pi=None, meta=False):
    if pi is None:
        pi = show_progress(os.path.basename(filename))
    ds = gdal.Open(filename)

    def it():
        # xsize = ds.RasterXSize
        nrows = ds.RasterYSize
        band = ds.GetRasterBand(1)
        row = band.ReadAsArray(0, 0, win_ysize=1)
        if band.GetNoDataValue() != get_nodata_value(row.dtype):
            print("WARNING: Incorrect nodata value: %s != %s" %
                  (band.GetNoDataValue(), get_nodata_value(row.dtype)))
        progress = -1
        for i in range(nrows):
            band.ReadAsArray(0, i, win_ysize=1, buf_obj=row)
            yield row.ravel()
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
