import sys
import collections
import numpy as np
sys.path.append('/home/rav/pygdal')
import raster


def load_subtree_size_from_dfs(filename):
    a, b, c, d = raster.load(filename, bands=4)
    e = np.ascontiguousarray(np.transpose((a, b), (1, 2, 0))).view(np.uint64)
    f = np.ascontiguousarray(np.transpose((c, d), (1, 2, 0))).view(np.uint64)
    result = 1 + (f - e) // 2
    print(collections.Counter(result.ravel()))
    return result


DX = [0]*129
DY = [0]*129
DX[1] = DX[2] = DX[128] = 1  # east
DX[8] = DX[16] = DX[32] = -1  # west
DY[2] = DY[4] = DY[8] = 1  # south
DY[32] = DY[64] = DY[128] = -1  # north


def main():
    subtree_size = load_subtree_size_from_dfs('mandoe-1m3-dfs.tif')
    dirs = raster.load('mandoe-1m3-dirs.tif')
    child_weight = np.zeros(dirs.shape, np.uint64)
    child_dir = np.zeros(dirs.shape, np.uint8)
    for i, row in raster.iterprogress(enumerate(dirs), n=len(dirs)):
        for j, dir in enumerate(row):
            if dir in (0, 255):
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            if child_weight[pi, pj] < subtree_size[i, j]:
                child_dir[pi, pj] = dir
                child_weight[pi, pj] = subtree_size[i, j]
    print('<ipeselection pos="0 0">')
    for i, row in raster.iterprogress(enumerate(dirs), n=len(dirs)):
        for j, dir in enumerate(row):
            if dir in (0, 255):
                continue
            if subtree_size[i, j] > 1:
                continue
            print('<path>')
            print('%s %s m' % (-i, j))
            pi = i + DY[dir]
            pj = j + DX[dir]
            while child_dir[pi, pj] == dir:
                print('%s %s l' % (-pi, pj))
                dir = dirs[pi, pj]
                if dir in (0, 255):
                    break
                pi += DY[dir]
                pj += DX[dir]
            print('</path>')
    print('</ipeselection>')


if __name__ == '__main__':
    main()
