'''
Helper functions for pygdal for easy row-by-row iteration of rasters.
'''

from __future__ import division

import os
import sys
from osgeo import gdal
import time
import functools
import contextlib

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


def iterprogress(iterable, pi=None, n=None):
    if n is None:
        n = len(iterable)
    if pi is None:
        pi = show_progress()
    for i, row in enumerate(iterable):
        yield row
        pi(i+1, n)


class ExitGenerator(Exception):
    pass


class ContextGenerator:
    def __init__(self, gen):
        self._gen = iter(gen)
        self._throw = self._gen.throw

    def __iter__(self):
        return self._gen

    def __next__(self):
        return next(self._gen)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._throw(ExitGenerator())
        except (ExitGenerator, StopIteration):
            pass
        else:
            raise RuntimeError("Generator didn't stop upon ExitGenerator")


def contextgenerator(gen_fn):
    @functools.wraps(gen_fn)
    def wrapper(*args, **kwargs):
        return ContextGenerator(gen_fn(*args, **kwargs))

    return wrapper


def iterrows(filename, pi=None, meta=False, buffer_rows=1, reverse=False):
    if pi is None:
        pi = show_progress(os.path.basename(filename))
    ds = gdal.Open(filename)

    @contextgenerator
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
        try:
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
        except ExitGenerator:
            pass
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


def write_raster_gen(filename, dtype, xsize, ysize):
    driver_name = 'GTiff'
    out_driver = gdal.GetDriverByName(driver_name)
    try:
        dtype_name = dtype.name
    except AttributeError:
        dtype_name = dtype.__name__
    dtypes = {
        'bool': gdal.GDT_Byte,
        'float32': gdal.GDT_Float32,
        'float64': gdal.GDT_Float64,
        'int32': gdal.GDT_Int32,
        'uint32': gdal.GDT_UInt32,
    }
    gdal_dtype = dtypes[dtype_name]

    nbands = 1

    assert xsize > 0
    assert ysize > 0

    options = ['%s=%s' % kv for kv in iteroptions(driver_name, dtype)]

    ds = out_driver.Create(filename, xsize, ysize, nbands, gdal_dtype, options)
    try:
        yield ds
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(np.float64(get_nodata_value(dtype)))
        yield band
        del band
        del ds
    except KeyboardInterrupt:
        print("Removing %r" % (filename,))
        os.remove(filename)
        raise


def write_raster_base(filename, dtype, f, f_ds, xsize, ysize):
    g = write_raster_gen(filename, dtype, xsize, ysize)
    ds = next(g)
    try:
        f_ds(ds)
    except BaseException as exn:
        g.throw(exn)
        raise
    band = next(g)
    try:
        f(band)
    except BaseException as exn:
        g.throw(exn)
        raise
    rest = list(g)  # Exhaust generator
    assert rest == []


@contextlib.contextmanager
def raster_writer_base(filename, dtype, meta):
    xsize = meta.RasterXSize
    ysize = meta.RasterYSize
    if xsize == 0 or ysize == 0:
        raise Exception("Invalid raster meta supplied (size %d×%d)" %
                        (xsize, ysize))
    g = write_raster_gen(filename, dtype, xsize, ysize)
    ds = next(g)
    ds.SetGeoTransform(meta.GetGeoTransform())
    ds.SetProjection(meta.GetProjection())
    band = next(g)
    try:
        yield band
    except BaseException as exn:
        g.throw(exn)
    rest = list(g)  # Exhaust generator
    assert rest == []
    del band
    del ds


def write_raster(filename, f, dtype, meta):
    '''Write single-band raster to filename by invoking f() on the band.

    See also raster_sink (for writing rows from a generator).
    '''
    with raster_writer_base(filename, dtype, meta) as band:
        f(band)


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


class RasterWriter:
    def __init__(self, filename, dtype, meta):
        self._writer = raster_writer_base(filename, dtype, meta)
        self._band = None

    def __enter__(self):
        self._band = self._writer.__enter__()
        self._current_row = 0
        return self

    def write_row(self, row):
        self._band.WriteArray(row.reshape(1, -1), 0, self._current_row)
        self._current_row += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._writer.__exit__(exc_type, exc_value, traceback)
        finally:
            self._writer = None


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


def load(filename, pi=None, bands=None):
    ds = gdal.Open(filename)
    if bands is not None:
        return tuple(ds.GetRasterBand(i+1).ReadAsArray(0, 0)
                     for i in range(bands))
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(0, 0)


def get_nodata_value(dtype):
    if dtype in (np.bool_, bool):
        return 2
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
    result = np.empty(shape, dtype=dtype)
    result[:] = get_nodata_value(dtype)
    return result


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
