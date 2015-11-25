from __future__ import division

import numpy as np
import raster


def main():
    filename = (
        '/home/rav/rasters/dk/' +
        'dhmbro-buildings-corrections-simpl-dfs-watersheds.tif')
    counts = np.zeros(40000000, dtype=np.uint32)
    counts, max_different = get_counts(raster.iterrows(filename), counts)
    print(max_different)
    np.savez_compressed('counts.npz', counts=counts)

    ind = np.argsort(counts)
    print(ind[-10:])
    print(counts[ind[-10:]])


def get_counts(rows, counts):
    """
    >>> get_counts([
    ...     [0, 0, 1, 1, 3, 5, -1],
    ...     [1, 2, 0, 0, 2, 4, -1],
    ... ], np.zeros(8))
    array([ 4.,  3.,  2.,  1.,  1.,  1.,  0.,  0.])
    """

    nodata = np.uint32(-1)
    max_different = 0
    for row in rows:
        row = np.asarray(row, dtype=np.uint32)
        row.sort()
        i = np.searchsorted(row, nodata)
        if i == 0:
            # All nodata
            continue
        row = row[:i]
        flag = np.concatenate(([True], row[1:] != row[:-1]))
        values = row[flag]
        idx = np.concatenate(np.nonzero(flag) + ([row.size],))
        c = np.diff(idx).astype(counts.dtype)
        counts[values] += c
        max_different = max(max_different, len(values))
    return counts, max_different


if __name__ == "__main__":
    main()
