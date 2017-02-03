#!/usr/bin/env python3
'''
Convert stupid "vectorized raster" format
(e.g. raster data in a vector file format)
into a proper raster.
'''

from __future__ import unicode_literals

# import ast
import osr
import gdal
import json
import argparse
from osgeo import ogr
import numpy as np
import raster


# def eval(s):
#     return ast.literal_eval(ast.parse(s, mode='eval'))


def eval_int(s):
    """
    >>> eval_int('4-2')
    2
    """
    return int(eval(s))


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('width', type=eval_int)
    parser.add_argument('height', type=eval_int)
    parser.add_argument('origin_x', type=eval_int)
    parser.add_argument('origin_y', type=eval_int)
    parser.add_argument('pixel_width', type=eval_int)
    parser.add_argument('pixel_height', type=eval_int)
    parser.add_argument('epsg', type=eval_int)
    parser.add_argument('--nbands', default=1, type=eval_int)
    args = parser.parse_args()

    driver = gdal.GetDriverByName(b"GTiff")
    ds = driver.Create(args.output, args.width, args.height, args.nbands,
                       gdal.GDT_Float32)

    ds.SetGeoTransform([args.origin_x, args.pixel_width, 0,
                        args.origin_y, 0, -args.pixel_height])
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(args.epsg)
    ds.SetProjection(sr.ExportToWkt())

    row = np.zeros((1, args.width), dtype=np.float32)
    band = ds.GetRasterBand(1)
    for i in range(args.height):
        band.WriteArray(row, 0, i)


def get_square(polygon):
    assert len(polygon) == 1
    vertices, = polygon

    assert len(vertices) == 5, len(vertices)
    assert vertices[0] == vertices[-1]
    xs, ys = zip(*vertices)
    assert len(set(xs)) == 2
    assert len(set(ys)) == 2
    x0 = min(xs)
    x1 = max(xs)
    # Note negative height
    y0 = max(ys)
    y1 = min(ys)
    return x0, y0, x1 - x0, y1 - y0


def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    ds = driver.Open(args.input, 0)
    assert ds is not None
    layer = ds.GetLayer()
    projection = layer.GetSpatialRef().ExportToWkt()
    features = [layer.GetFeature(i) for i in range(layer.GetFeatureCount())]
    features = [json.loads(f.ExportToJson()) for f in features]
    assert all(f['type'] == 'Feature' for f in features)
    assert all(f['geometry']['type'] == 'Polygon' for f in features)
    property_keys = frozenset(features[0]['properties'].keys())
    assert all(frozenset(f['properties'].keys()) == property_keys
               for f in features)
    property_keys = sorted(property_keys)

    squares = [get_square(f['geometry']['coordinates']) for f in features]
    xs, ys, widths, heights = zip(*squares)
    assert len(set(widths)) == 1
    assert len(set(heights)) == 1
    cell_width, = set(widths)
    cell_height, = set(heights)
    assert cell_width > 0
    assert cell_height < 0
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    origin_x = xs.min()
    # Note negative height
    origin_y = ys.max()

    geo_transform = [origin_x, cell_width, 0, origin_y, 0, cell_height]

    col_index = np.round((xs - origin_x) / cell_width).astype(np.int)
    row_index = np.round((ys - origin_y) / cell_height).astype(np.int)
    raster_width = col_index.max() + 1
    raster_height = row_index.max() + 1

    r = raster.empty((raster_height, raster_width), np.float32)
    for k in property_keys:
        r[row_index, col_index] = [
            feature['properties'][k] for feature in features]
        raster.write_generated_raster(
            args.output + k + '.tif', r, projection, geo_transform)


if __name__ == "__main__":
    main2()
