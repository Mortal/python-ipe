#!/usr/bin/env python3
import re
import argparse
import itertools
import collections
from raster import (
    iterrows, write_raster_base, peek_dtype,
    dummy_progress, show_progress, iterprogress,
)


Spec = collections.namedtuple('Spec', 'width height left top')


def parse_spec(s):
    mo = re.match(r'^(\d+)x(\d+)\+(\d+)\+(\d+)$', s)
    if not mo:
        raise ValueError('format: WIDTHxHEIGHT+LEFT+TOP')
    return Spec(*map(int, mo.groups()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--spec', required=True,
                        help='WIDTHxHEIGHT+LEFT+TOP', type=parse_spec)
    args = parser.parse_args()
    width, height, left, top = args.spec

    meta, rows = iterrows(args.input, meta=True, pi=dummy_progress)

    input_geo = meta.GetGeoTransform()
    origin_x, pixel_width, zero, origin_y, zero_, pixel_height = input_geo
    assert zero == zero_ == 0
    output_geo = [origin_x + left * pixel_width, pixel_width, 0,
                  origin_y + top * pixel_height, 0, pixel_height]

    def handle_ds(ds):
        ds.SetGeoTransform(output_geo)
        ds.SetProjection(meta.GetProjection())

    dtype, rows = peek_dtype(rows)
    rows = itertools.islice(rows, top, top + height)
    rows = (row[left:left+width] for row in rows)
    rows = iterprogress(rows, show_progress(args.output), height)

    def handle_band(band):
        for i, row in enumerate(rows):
            band.WriteArray(row.reshape(1, -1), 0, i)

    write_raster_base(args.output, dtype, handle_band, handle_ds,
                      width, height)


if __name__ == '__main__':
    main()
