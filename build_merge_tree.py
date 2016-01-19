from __future__ import division

from itertools import izip as zip
from itertools import tee

import numpy as np
import raster

from raster import get_nodata_value


def window(filename, meta=False):
    meta, rows = raster.iterrows(filename, meta=True)
    it = raster.window(rows)
    if meta:
        return meta, it
    else:
        return it


class Slices(object):
    def __getitem__(self, slices):
        return slices


slices = Slices()


def main():
    elev_name = '/data/rasters/20150224/bug8.tif'
    rank_name = '/data/rasters/20150224/bug8-ranks.tif'
    output_name = '/data/rasters/degrees.tif'
    meta, elev = raster.iterrows(elev_name, meta=True)
    rank = raster.iterrows(rank_name, pi=raster.dummy_progress)

    d = degrees_logged(elev, rank)
    raster.raster_sink(output_name, d, np.uint32, meta)


def elev_rank_lt(e1, r1, e2, r2):
    eq = e1 == e2
    lt = e1 < e2  # generally use elevations to decide less-than
    rlt = r1 < r2
    lt[eq] = rlt[eq]  # but use ranks when elevs are equal
    return lt


def elev_rank_le(e1, r1, e2, r2):
    return ~elev_rank_lt(e2, r2, e1, r1)


def degrees(elev, rank):
    """Compute vertex degrees of (N, M) terrain

    Parameters
    ----------
    elev : (M,) float N-iterable
    rank : (M,) int N-iterable

    Returns
    -------
    deg : (M,) int N-iterable
        Entry [i][j] is 0 for source/sink, 1 for regular, d for d-saddle
    """

    for (ae, be, ce), (ar, br, cr) in raster.window(elev, rank):
        # cmps[i, j, k] == True <=> center cell k is above neighbor [i, j]
        cmps = np.zeros((3, 3, len(be)))
        isdata = be != get_nodata_value(be.dtype)

        for i, ie, ir in zip(cmps, (ae, be, ce), (ar, br, cr)):
            idx_other = slices[:-1, :, 1:]  # ie/ir indices of 'other' operand
            # idx_self = slices[1:, :, :-1]  # be/br indices of 'self' operand
            for ii, j1 in zip(i, idx_other):
                e1 = ie[j1]
                r1 = ir[j1]
                if j1.stop == -1:
                    # Comparing with left neighbor; add nodata on the left
                    e1 = np.concatenate(([get_nodata_value(e1.dtype)], e1))
                    r1 = np.concatenate(([get_nodata_value(r1.dtype)], r1))
                elif j1.start == 1:
                    # Comparing with right neighbor; add nodata on the right
                    e1 = np.concatenate((e1, [get_nodata_value(e1.dtype)]))
                    r1 = np.concatenate((r1, [get_nodata_value(r1.dtype)]))
                # (e1,r1) is the [i, j] neighbor
                # (be,br) is self

                # Raster order determines if we should use lt or le
                if ie is ae or (ie is be and j1.start is None):
                    # neighbor is raster-less-than self,
                    # so it suffices for it to be topo-less-or-equal
                    ii[:] = elev_rank_le(e1, r1, be, br)
                else:
                    # neighbor is raster-greater-than self,
                    # so it must be topo-strictly-less-than
                    ii[:] = elev_rank_lt(e1, r1, be, br)
        cmps = np.asarray([
            cmps[0, 0],
            cmps[1, 0],
            cmps[2, 0],
            cmps[2, 1],
            cmps[2, 2],
            cmps[1, 2],
            cmps[0, 2],
            cmps[0, 1],
            cmps[0, 0],
        ])
        cmp_same = cmps[:-1, :] == cmps[1:, :]
        cmp_diffs = np.sum(~cmp_same, axis=0)
        assert np.all(cmp_diffs % 2 == 0)
        yield cmp_diffs / 2


def degrees_logged(elev, rank):
    """Wrapper around degrees(elev, rank), outputting some statistics."""
    saddles = 0
    extremes = sinks = 0
    regulars = 0
    cells = 0
    elevdata = elevnodata = 0

    elev1, elev2 = tee(elev)
    rank1, rank2 = tee(rank)
    for e_row, r_row, d_row in zip(elev1, rank1, degrees(elev2, rank2)):
        cells += len(e_row)
        isdata = e_row != get_nodata_value(e_row.dtype)
        elevdata += isdata.sum()
        elevnodata += (~isdata).sum()
        extreme = isdata & (d_row == 0)
        extremes += extreme.sum()
        sink_right = elev_rank_le(e_row[:-1], r_row[:-1], e_row[1:], r_row[1:])
        sink_left = elev_rank_lt(e_row[1:], r_row[1:], e_row[:-1], r_row[:-1])
        sink = (
            np.concatenate((sink_right, [False])) &
            np.concatenate(([False], sink_left)) & extreme)
        sinks += sink.sum()
        source = extreme & (~sink)
        regular = isdata & (d_row == 1)
        regulars += regular.sum()
        saddle = isdata & (d_row > 1)
        saddles += saddle.sum()
        d_row[saddle] += 1
        d_row[source] = 2
        yield d_row
    print("%s cells" % cells)
    print("%s data, %s nodata" % (elevdata, elevnodata))
    print("Regular vertices: %s" % regulars)
    print("Extremes: %s" % extremes)
    print("Sinks: %s" % sinks)
    print("Saddles: %s" % saddles)


if __name__ == "__main__":
    main()
