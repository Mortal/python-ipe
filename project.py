#!/usr/bin/env python3
'''
Program to convert between Google Maps lat/lon coordinates and raster local
coordinates using pygdal (to read GeoTIFF projection info) and pyproj.

The two required command line arguments are the source and destination projection
which can be 'gmaps' or a GeoTIFF filename.

On standard input, give lines consisting of source coordinates (if source is gmaps)
or source column/row indices (if source is a file).
The output is always coordinates (not column/row indices).
'''

from __future__ import unicode_literals

import ast
import sys
import gdal
import pyproj
import raster
import argparse
import subprocess


def eval(s):
    return ast.literal_eval(ast.parse(s, mode='eval'))


def wkt_to_proj(s):
    return eval(subprocess.check_output(
        ('gdalsrsinfo', '-o', 'proj4', s),
        universal_newlines=True))


def get_projection(filename):
    prep = None
    if filename == 'gmaps':
        s = '+init=epsg:3857'
    else:
        ds = gdal.Open(filename)
        rows = ds.RasterYSize
        columns = ds.RasterXSize
        x0, dx, _1, y0, _2, dy = ds.GetGeoTransform()
        assert _1 == _2 == 0
        s = wkt_to_proj(ds.GetProjection())

        def prep(x, y):
            return (x0 + dx * x, y0 + dy * y)

    return pyproj.Proj(s), prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=get_projection)
    parser.add_argument('destination', type=get_projection)
    args = parser.parse_args()

    sproj, sprep = args.source
    dproj, dprep = args.destination

    for line in sys.stdin:
        x, y = line.split()
        x, y = float(x), float(y)
        if sprep is not None:
            x, y = sprep(x, y)
        print(x, y, pyproj.transform(sproj, dproj, x, y))


if __name__ == "__main__":
    main()
