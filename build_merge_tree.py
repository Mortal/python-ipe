from __future__ import division, print_function

import collections
from itertools import izip as zip
from itertools import imap as map
from itertools import tee, groupby

import numpy as np
import raster

from raster import get_nodata_value, add_nodata_row


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


def elev_rank_lt(e1, r1, e2, r2):
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

    eq = e1 == e2
    lt = e1 < e2  # generally use elevations to decide less-than
    rlt = r1 < r2
    lt[eq] = rlt[eq]  # but use ranks when elevs are equal
    return lt


def elev_rank_le(e1, r1, e2, r2):
    return ~elev_rank_lt(e2, r2, e1, r1)


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

    n_elev = n_rank = cmps = cmps2 = None
    for (ae, be, ce), (ar, br, cr) in raster.window(elev, rank):
        if n_elev is None:
            n_elev = np.zeros((3, 3, len(be)), dtype=be.dtype)
            n_elev += get_nodata_value(be.dtype)
        if n_rank is None:
            n_rank = np.zeros((3, 3, len(br)), dtype=br.dtype)
            n_rank += get_nodata_value(br.dtype)
        if cmps is None:
            cmps = np.zeros((3, 3, len(be)), dtype=np.bool)
        neighbors(ae, be, ce, n_elev)
        neighbors(ar, br, cr, n_rank)
        for i in range(3):
            for j in range(3):
                if (i, j) < (1, 1):
                    cmps[i, j, :] = elev_rank_le(
                        n_elev[i, j], n_rank[i, j], be, br)
                else:
                    cmps[i, j, :] = elev_rank_lt(
                        n_elev[i, j], n_rank[i, j], be, br)
        if cmps2 is None:
            cmps2 = cmps[
                [0, 1, 2, 2, 2, 1, 0, 0, 0],
                [0, 0, 0, 1, 2, 2, 2, 1, 0],
            ]
        else:
            cmps2[:] = cmps[
                [0, 1, 2, 2, 2, 1, 0, 0, 0],
                [0, 0, 0, 1, 2, 2, 2, 1, 0],
            ]
        cmp_same = cmps2[:-1, :] == cmps2[1:, :]
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


def negative_saddles(elev, rank, wsheds):
    elev1, elev2 = tee(elev)
    deg_it = degrees(elev1, rank)
    ws_it = raster.window(wsheds)
    data = enumerate(zip(*map(add_nodata_row, (elev2, deg_it, ws_it))))
    result = []
    saddle = []
    for j, (z, deg, (wa, wb, wc)) in data:
        s = deg > 1
        orig_idxs = s.nonzero()[0]
        s_n = len(orig_idxs)
        nodata = get_nodata_value(wb.dtype)
        neighbor_watersheds = np.zeros((s_n, 9), dtype=wb.dtype)
        for c, r in enumerate((wa, wb, wc)):
            i = slice(1, None) if s[0] else slice(None, None)
            neighbor_watersheds[i, 3*c] = r[:-1][s[1:]]
            neighbor_watersheds[:, 3*c+1] = r[s]
            i = slice(None, -1) if s[-1] else slice(None, None)
            neighbor_watersheds[i, 3*c+2] = r[1:][s[:-1]]
        neighbor_watersheds[neighbor_watersheds == nodata] = 0
        neighbor_watersheds.sort()
        diff = neighbor_watersheds[:, :-1] != neighbor_watersheds[:, 1:]
        diff = np.c_[np.repeat(True, len(neighbor_watersheds)), diff]
        ndiff = diff.sum(axis=1)

        for i in (ndiff > 1).nonzero()[0]:
            w = neighbor_watersheds[i][diff[i]].tolist()
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
