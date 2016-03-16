import raster
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    meta, rows = iterrows(args.input, meta=True, reverse=True)
    nodata = raster.get_nodata_value(np.float32)
    closest_y = -np.ones(meta.RasterXSize)
    closest_val = np.zeros(meta.RasterXSize)
    out_dd = np.zeros(meta.RasterXSize, dtype=np.uint32)
    out_y = np.zeros(meta.RasterXSize, dtype=np.uint32)
    xindices = np.arange(meta.RasterXSize)
    for i, row in enumerate(rows):
        is_nodata = row == nodata
        is_data = ~is_nodata
        closest_y[is_data] = i
        closest_val[is_data] = row[is_data]
        for j in is_nodata.nonzero()[0]:
            dy = (closest_y - i)**2
            dx = np.abs(xindices - j)**2
            closest = np.argmin(dy + dx)


if __name__ == "__main__":
    main()
