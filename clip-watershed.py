import argparse
from itertools import izip as zip

import numpy as np

import raster


def input_range(s):
    parts = s.split(',')
    result = []
    for p in parts:
        args = [int(v) for v in p.split(':')]
        if len(args) > 1:
            r = range(*args)
            if len(parts) == 1:
                return r
            else:
                result.extend(list(r))
        else:
            result.append(args[0])
    return set(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--input-watersheds', required=True)
    parser.add_argument('-i', '--input-target', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-v', '--values', type=input_range, required=True)
    parser.add_argument('-b', '--boundary', type=int, default=1)
    args = parser.parse_args()

    boundary = args.boundary

    if args.input_watersheds == args.input_target:
        r = raster.iterrows(args.input_watersheds)
        rows = ((row, row) for row in r)
    else:
        r1 = raster.iterrows(args.input_watersheds)
        r2 = raster.iterrows(args.input_target, buffer_rows=1 + boundary)
        rows = zip(r1, r2)

    prev_rows = []
    output_rows = []
    needle = np.array(list(args.values)).reshape((1, -1))
    after_rows = 0
    for row1, row2 in rows:
        eq = row1.reshape((-1, 1)) == needle
        nz = eq.any(axis=1).nonzero()[0]
        if len(nz):
            if boundary:
                output_rows.extend(
                    [(nz[0], nz[-1] + 1, np.array(row))
                     for row in prev_rows[-boundary:]])
                prev_rows = []
                after_rows = boundary
            output_rows.append((nz[0], nz[-1] + 1, np.array(row2)))
        elif after_rows > 0:
            after_rows -= 1
            p = output_rows[-1]
            output_rows.append((p[0], p[1], np.array(row2)))
        else:
            prev_rows.append(row2)

    if not output_rows:
        raise SystemExit("%s not found" % (needle,))
    i = min(x[0] for x in output_rows)
    j = max(x[1] for x in output_rows)
    print("Write %dx%d raster" % (len(output_rows), j - i))
    raster.write_generated_raster(
        args.output,
        [x[2][i:j] for x in output_rows])


if __name__ == "__main__":
    main()
