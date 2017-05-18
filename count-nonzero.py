import argparse
import numpy as np
from raster import (
    iterrows, show_progress, dummy_progress, is_data, raster_sink,
)


def count_nonzero(rasters_rows, dtype):
    output_row = None
    for rows in rasters_rows:
        output_row = np.sum([is_data(row) for row in rows], axis=0,
                            dtype=dtype, out=output_row)
        yield output_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('filenames', nargs='+')
    args = parser.parse_args()
    base, *rest = args.filenames
    meta, base_rows = iterrows(base, meta=True)
    rest_rows = (iterrows(f, dummy_progress) for f in rest)
    raster_zip = zip(base_rows, *rest_rows)
    dtype = np.uint32
    raster_sink(args.output, count_nonzero(raster_zip, dtype), dtype, meta)


if __name__ == '__main__':
    main()
