from __future__ import division, print_function

import collections
from itertools import izip as zip
from itertools import imap as map
from itertools import tee, groupby

import numpy as np
import raster

from raster import get_nodata_value, add_nodata_row, peek_row


class Slices(object):
    def __getitem__(self, slices):
        return slices


slices = Slices()


def main():
    elev_name = '/data/rasters/20150224/bug8.tif'
    rank_name = '/data/rasters/20150224/bug8-ranks.tif'
    wsheds_name = '/data/rasters/20150224/bug8-watersheds.tif'
    output_name = '/data/rasters/degrees.tif'
    meta, elev = raster.iterrows(elev_name, meta=True)
    rank = raster.iterrows(rank_name, pi=raster.dummy_progress)
    wsheds = raster.iterrows(wsheds_name, pi=raster.dummy_progress)

    # d = degrees_logged(elev, rank)
    # raster.raster_sink(output_name, d, np.uint32, meta)

    ns = negative_saddles(elev, rank, wsheds)
    edges, saddles = zip(*ns)
    zs, loc_i, loc_j = zip(*saddles)
    zs = np.asarray(zs)
    locs = np.c_[loc_i, loc_j]
    # np.savez_compressed('build_merge_tree.npz', edges=edges, z=zs, i=i, j=j)

    # o = np.load('build_merge_tree.npz')
    # edges = o['edges']
    # zs = o['z']
    # locs = np.c_[o['i'], o['j']]
    edges = np.asarray(edges)
    idx = np.argsort(zs)
    rep = dict()
    depressions = dict()
    points = []
    for (w1, w2), z, loc in zip(edges[idx], zs[idx], locs[idx]):
        saddle = (w1, w2, z, loc)
        rep.setdefault(w1, w1)
        while w1 != rep[w1]:
            w1 = rep[w1] = rep[rep[w1]]
        rep.setdefault(w2, w2)
        while w2 != rep[w2]:
            w2 = rep[w2] = rep[rep[w2]]
        if w1 != w2:
            d1 = depressions.pop(w1, (1, w1))
            d2 = depressions.pop(w2, (1, w2))
            rep[w2] = w1
            depressions[w1] = (d1[0] + d2[0], saddle, d1, d2)
            points.append((loc.tolist(), d1[0] + d2[0]))
        else:
            # print("Positive saddle %s" % (saddle,))
            points.append((loc.tolist(), 0))
    # print(depressions.keys())
    print(len(depressions))
    points.sort()
    raster.points_to_raster(output_name, points, np.uint32, meta)


def elev_rank_lt(e1, r1, e2, r2, out=None, buf=None):
    """Decide if a cell is below another cell given elevations and ranks.

    Parameters
    ----------
    e1, e2 : float ndarrays of same shape
    r1, r2 : integer ndarray, same shape as e1/e2 or empty
        Elevations and ranks of cells to compare.
        If r1.size == 0, ranks are assumed to be all equal.

    Returns
    -------
    lt : bool ndarray, same shape as input
        lt[I] == True <=> e1[I] < e2[I] or (e1[I] == e2[I] and r1[I] < r2[I])
    """

    e1, e2 = np.asarray(e1), np.asarray(e2)
    r1, r2 = np.asarray(r1), np.asarray(r2)
    assert e1.shape == e2.shape
    assert r1.shape == r2.shape
    if not r1.size:
        return e1 < e2
    assert r1.shape == e1.shape

    if buf is None:
        eq = e1 == e2
    else:
        eq = np.equal(e1, e2, out=buf[0])
    if out is None:
        lt = e1 < e2  # generally use elevations to decide less-than
    else:
        lt = np.less(e1, e2, out=out)
    if buf is None:
        rlt = r1 < r2
    else:
        rlt = np.less(r1, r2, out=buf[1])
    lt[eq] = rlt[eq]  # but use ranks when elevs are equal
    return lt


def elev_rank_le(e1, r1, e2, r2, buf=None, out=None):
    if out is None:
        return ~elev_rank_lt(e2, r2, e1, r1, buf=buf)
    else:
        elev_rank_lt(e2, r2, e1, r1, out=out, buf=buf)
        return np.invert(out, out=out)


def neighbors(a, b, c, out):
    out[0, 0, 1:] = a[:-1]
    out[0, 1, :] = a[:]
    out[0, 2, :-1] = a[1:]
    out[1, 0, 1:] = b[:-1]
    out[1, 1, :] = b[:]
    out[1, 2, :-1] = b[1:]
    out[2, 0, 1:] = c[:-1]
    out[2, 1, :] = c[:]
    out[2, 2, :-1] = c[1:]
    return out


def neighbors_masked(a, b, c, m, out):
    assert a.shape == b.shape == c.shape == m.shape
    assert out.shape == (3, 3, len(m.nonzero()[0]))
    i = slice(1, None) if m[0] else slice(None, None)
    j = slice(None, -1) if m[-1] else slice(None, None)
    out[0, 0, i] = a[:-1][m[1:]]
    out[0, 1, :] = a[m]
    out[0, 2, j] = a[1:][m[:-1]]
    out[1, 0, i] = b[:-1][m[1:]]
    out[1, 1, :] = b[m]
    out[1, 2, j] = b[1:][m[:-1]]
    out[2, 0, i] = c[:-1][m[1:]]
    out[2, 1, :] = c[m]
    out[2, 2, j] = c[1:][m[:-1]]


def take_output(a, indices, out):
    for i, j in zip(zip(*indices), out):
        j[:] = a[i]


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

    row, elev = peek_row(elev)
    n_elev = np.zeros((3, 3, len(row)), dtype=row.dtype)
    n_elev += get_nodata_value(n_elev.dtype)
    row, rank = peek_row(rank)
    n_rank = np.zeros((3, 3, len(row)), dtype=row.dtype)
    n_rank += get_nodata_value(n_rank.dtype)
    cmps = np.zeros((3, 3, len(row)), dtype=np.bool)
    cmps2 = np.zeros((9, len(row)), dtype=np.bool)
    cmp_same = np.zeros((8, len(row)), dtype=np.bool)
    cmp_diffs = np.zeros(len(row), dtype=np.uint8)
    elev_rank_buf = np.zeros((2, len(row)), dtype=np.bool)
    for (ae, be, ce), (ar, br, cr) in raster.window(elev, rank):
        neighbors(ae, be, ce, out=n_elev)
        neighbors(ar, br, cr, out=n_rank)
        for i in range(3):
            for j in range(3):
                if (i, j) < (1, 1):
                    elev_rank_le(
                        n_elev[i, j], n_rank[i, j], be, br,
                        out=cmps[i, j], buf=elev_rank_buf)
                else:
                    elev_rank_lt(
                        n_elev[i, j], n_rank[i, j], be, br,
                        out=cmps[i, j], buf=elev_rank_buf)
        take_output(
            cmps,
            [
                [0, 1, 2, 2, 2, 1, 0, 0, 0],
                [0, 0, 0, 1, 2, 2, 2, 1, 0],
            ],
            out=cmps2)
        np.equal(cmps2[:-1, :], cmps2[1:, :], out=cmp_same)
        np.sum(~cmp_same, axis=0, out=cmp_diffs)
        assert np.all(cmp_diffs % 2 == 0)
        yield np.divide(cmp_diffs, 2, out=cmp_diffs)


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


def negative_saddles(elev, rank, wsheds):
    elev1, elev2 = tee(elev)
    row, wsheds = peek_row(wsheds)
    deg_it = degrees(elev1, rank)
    ws_it = raster.window(wsheds)
    data = enumerate(zip(*map(add_nodata_row, (elev2, deg_it, ws_it))))
    result = []
    saddle = []
    neighbor_watersheds = np.zeros((9, len(row)), dtype=row.dtype)
    for j, (z, deg, (wa, wb, wc)) in data:
        s = deg > 1
        orig_idxs = s.nonzero()[0]
        s_n = len(orig_idxs)
        nodata = get_nodata_value(wb.dtype)
        neighbors_masked(wa, wb, wc, s,
                         out=neighbor_watersheds[:, :s_n].reshape((3, 3, -1)))
        neighbor_watersheds[neighbor_watersheds == nodata] = 0
        neighbor_watersheds.sort(axis=0)
        diff = neighbor_watersheds[:-1, :s_n] != neighbor_watersheds[1:, :s_n]
        ts = np.repeat(True, s_n)
        diff = np.r_[ts.reshape((1, s_n)), diff]
        ndiff = diff.sum(axis=0)

        for i in (ndiff > 1).nonzero()[0]:
            w = neighbor_watersheds[:, i][diff[:, i]].tolist()
            assert len(w) > 1
            for k1 in range(len(w)):
                for k2 in range(k1 + 1, len(w)):
                    e = (w[k1], w[k2])
                    v = (z[orig_idxs[i]], j, orig_idxs[i])
                    saddle.append((e, v))
        if saddle:
            saddle.sort()
            saddle2 = []
            searches = []
            res = [None]
            for e, gr in groupby(saddle, key=lambda x: x[0]):
                searches.append(e)
                res.append(min(gr))
            res = np.array(res, dtype=np.object)[1:]
            searches = np.asarray(searches)
            active = np.unique([wb, wc])
            if active[0] == 0:
                active = active[1:]
            indices = np.searchsorted(active, searches)
            indices[indices == len(active)] = 0
            found = (active[indices] == searches).any(axis=1)
            saddle = res[found].tolist()
            i = len(result)
            result.extend(res[~found].tolist())
    return result


if __name__ == "__main__":
    main()
