import argparse
from raster import iterrows, is_data
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    print(sum(np.sum(row[is_data(row)]) for row in iterrows(args.filename)))


if __name__ == '__main__':
    main()
