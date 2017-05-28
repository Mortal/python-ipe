'''
Helper functions for pygdal for easy row-by-row iteration of rasters.
'''

from __future__ import division

import os
import sys
import gdal
import time

from itertools import chain

import numpy as np

PY2 = sys.hexversion < 0x03000000
if PY2:
    from itertools import izip as zip  # noqa


DESCRIPTION = '''\
Helper functions for pygdal for easy row-by-row iteration of rasters.

Simple usage::

    from raster import iterrows
    for row in iterrows('georaster.tif'):
        print(row.mean())  # row is a NumPy array

Supports automatic progress indication when reading with iterrows;
pass pi=raster.dummy_progress to disable.

Read the source code for more information.
'''.rstrip()


gdal.UseExceptions()


def show_progress(name="", more_steps=1):
    # start time, current time, step per time, next update step, steps per display, next display step
    t = [0, 0, 0, 0, 1, 0]
    recalc_every = 100
    update_every = 0.1
    def pi(i, n):
        if i < t[5]:
            return
        if t[0] == 0:
            t[0] = time.time()
            t[3] = i + recalc_every
        elif i >= t[3]:
            t[1] = time.time()
            t[2] = i / (t[1] - t[0])
            t[3] = i + recalc_every
            t[4] = int(update_every * t[2])
        t[5] = min(n, i + t[4])
        output_time = ((t[2] and (more_steps * n - i) / t[2])
                       if i < n else time.time() - t[0])
        output_speed = '%g' % t[2]
        sys.stderr.write("\r\x1B[K%3d%% %s %12d/%d %-7s %.2f" %
                         (i * 100 / n, name, i, n, output_speed, output_time))
        if i == n:
            sys.stderr.write('\n')
        sys.stderr.flush()

    return pi


def dummy_progress(i, n):
    pass


def iterprogress(iterable, pi, n):
    for i, row in enumerate(iterable):
        yield row
        pi(i+1, n)


def iterrows(filename, pi=None, meta=False, buffer_rows=1, reverse=False):
    if pi is None:
        pi = show_progress(os.path.basename(filename))
    ds = gdal.Open(filename)

    def it(ds):
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
                src_row = i + k
                if reverse:
                    src_row = nrows - src_row - 1
                band.ReadAsArray(0, src_row, win_ysize=1, buf_obj=row[k:k+1])
                yield row[k]
                p = (i + 1) * 1000 // nrows
                if p > progress and i + 1 < nrows:
                    progress = p
                    pi(i + 1, nrows)
        pi(nrows, nrows)
        # Without this del we get
        # "SystemError: <built-in function delete_Dataset> returned a result with an error set"
        del ds

    if meta:
        return ds, it(ds)
    else:
        return it(ds)


def iteroptions(driver, dtype, compress=True):
    assert driver == 'GTiff'
    yield ('BIGTIFF', 'IF_SAFER')
    if dtype == np.bool:
        yield ('NBITS', '1')
    # if dtype == tribool:
    #     yield ('NBITS', '2')
    yield ('SPARSE_OK', 'TRUE')
    if compress:
        yield ('COMPRESS', 'DEFLATE')


def write_raster_base(filename, dtype, f, f_ds, xsize, ysize):
    driver_name = 'GTiff'
    out_driver = gdal.GetDriverByName(driver_name)
    try:
        dtype_name = dtype.name
    except AttributeError:
        dtype_name = dtype.__name__
    gdal_dtype = gdal.GetDataTypeByName(dtype_name)

    nbands = 1

    assert xsize > 0
    assert ysize > 0

    options = ['%s=%s' % kv for kv in iteroptions(driver_name, dtype)]

    ds = out_driver.Create(filename, xsize, ysize, nbands, gdal_dtype, options)
    try:
        f_ds(ds)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(np.float64(get_nodata_value(dtype)))
        f(band)
    except KeyboardInterrupt:
        print("Removing %r" % (filename,))
        os.remove(filename)
        raise


def write_raster(filename, f, dtype, meta):
    '''Write single-band raster to filename by invoking f() on the band.

    See also raster_sink (for writing rows from a generator).
    '''
    xsize = meta.RasterXSize
    ysize = meta.RasterYSize
    def f_ds(ds):
        ds.SetGeoTransform(meta.GetGeoTransform())
        ds.SetProjection(meta.GetProjection())

    return write_raster_base(filename, dtype, f, f_ds, xsize, ysize)


def write_generated_raster(filename, r, projection=None, geo_transform=None):
    def f_ds(ds):
        if projection is not None:
            ds.SetProjection(projection)
        if geo_transform is not None:
            ds.SetGeoTransform(geo_transform)

    def f(band):
        band.WriteArray(r)

    r = np.asarray(r)
    write_raster_base(filename, r.dtype, f, f_ds, r.shape[1], r.shape[0])


def raster_sink(filename, iterable, dtype, meta):
    '''Write single-band raster row-by-row to filename.'''
    def f(band):
        for i, row in enumerate(iterable):
            band.WriteArray(row.reshape(1, -1), 0, i)

    return write_raster(filename, f, dtype, meta)


def points_to_raster(filename, iterable, dtype, meta):
    def f():
        current_row = 0
        row = np.zeros(meta.RasterXSize, dtype=dtype)
        row[:] = get_nodata_value(dtype)
        for loc, z in iterable:
            while current_row < loc[0]:
                yield row
                row[:] = get_nodata_value(dtype)
                current_row += 1
            row[loc[1]] = z
        while current_row < meta.RasterYSize:
            yield row
            row[:] = get_nodata_value(dtype)
            current_row += 1

    return raster_sink(filename, f(), dtype, meta)


def load(filename, pi=None):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(0, 0)


def get_nodata_value(dtype):
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).min


def is_nodata(v):
    return v == get_nodata_value(v.dtype)


def is_data(v):
    return v != get_nodata_value(v.dtype)


def nodata_like(row):
    row = np.asarray(row)
    return np.zeros_like(row) + get_nodata_value(row.dtype)


def empty(shape, dtype):
    return np.zeros(shape, dtype=dtype) + get_nodata_value(dtype)


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


def peek_row(iterable):
    row = next(iterable)
    return row, chain([row], iterable)


def peek_dtype(iterable):
    '''Peek the dtype of the first row of iterable.

    Consumes iterable and returns (dtype, iterable).'''

    row, iterable = peek_row(iterable)
    return row.dtype, iterable
