#!/usr/bin/env python3
'''
Given two watershed rasters for the same terrain,
find out which watershed IDs correspond to each other.
'''

from __future__ import division, print_function

import argparse

import numpy as np

import raster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename1')
    parser.add_argument('filename2')
    args = parser.parse_args()

    raster1 = raster.load(args.filename1)
    raster2 = raster.load(args.filename2)
    pairs = []
    for r1, r2 in zip(raster1, raster2):
        ziprow = np.c_[r1, r2].view(np.int64).ravel()
        pairs.append(np.unique(ziprow))
    pairs = np.unique(np.concatenate(pairs)).reshape((-1, 1)).view(np.int32)
    print(len(pairs))
    print(len(np.unique(pairs[:, 0])))
    print(len(np.unique(pairs[:, 1])))


if __name__ == "__main__":
    main()
