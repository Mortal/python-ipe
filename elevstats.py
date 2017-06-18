import os
import argparse

import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import mplcache
from raster import iterrows, window, is_data, peek_row, empty, get_nodata_value, is_nodata
import numpy as np


def iterneighbors(iterable):
    row, iterable = peek_row(iterable)
    ncols = len(row)
    buffer = empty((8, ncols), dtype=row.dtype)

    NW, N, NE, E, SE, S, SW, W = range(8)

    for above, cur, below in window(iterable):
        buffer[NW, 1:] = above[:-1]
        buffer[N, :] = above[:]
        buffer[NE, :-1] = above[1:]
        buffer[E, :-1] = cur[1:]
        buffer[SE, :-1] = below[1:]
        buffer[S, :] = below[:]
        buffer[SW, 1:] = below[:-1]
        buffer[W, 1:] = cur[:-1]
        yield cur.reshape(1, ncols), buffer


def iterelevstats(iterable):
    for row, neighbors in iterneighbors(iterable):
        b = is_data(neighbors) & is_data(row)
        diff = neighbors - row
        diff[~b] = np.inf
        lowest = np.min(diff, axis=0, keepdims=True)
        yield (row[is_data(row)],
               -lowest[(lowest < 0) & is_data(row)],
               diff[b & (diff > 0)])


def elevstats(filename):
    iterable = iterrows(filename, buffer_rows=3)
    cells = []
    descent = []
    edges = []
    for r, l, e in iterelevstats(iterable):
        cells.append(r)
        descent.append(l)
        edges.append(e)
    cells = np.concatenate(cells)
    descent = np.concatenate(descent)
    edges = np.concatenate(edges)
    ps = np.arange(101)
    return (
        cells.mean(),
        np.percentile(cells, ps),
        np.percentile(descent, ps),
        np.percentile(edges, ps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    base, ext = os.path.splitext(args.filename)
    zmean, z_stats, descent_stats, edge_stats = elevstats(args.filename)

    print('Heights in range [%g, %g], avg %g' % (z_stats[0], z_stats[-1], zmean))
    z_span = z_stats[50:] - z_stats[50::-1]
    print('Height span: 50%%=%g 80%%=%g 95%%=%g max=%g' % tuple(z_span[[25, 40, 47, 50]]))
    ps = [50, 80, 95, 100]
    print('Steepest descent edges: 50%%=%g 80%%=%g 95%%=%g max=%g' % tuple(descent_stats[ps]))
    print('All edges: 50%%=%g 80%%=%g 95%%=%g max=%g' % tuple(edge_stats[ps]))

    all_ps = np.arange(101)
    fig, ax = plt.subplots()
    ax.plot(all_ps, z_stats)
    ax.grid()
    mplcache.savefig(fig, base + '_z.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(all_ps, descent_stats)
    ax.set_ylim(0, descent_stats[-2])
    ax.grid()
    mplcache.savefig(fig, base + '_descent.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(all_ps, edge_stats)
    ax.set_ylim(0, edge_stats[-2])
    ax.grid()
    mplcache.savefig(fig, base + '_edge.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
