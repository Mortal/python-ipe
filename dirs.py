import sys
import zlib
import base64
import collections
import numpy as np
sys.path.append('/home/rav/pygdal')
import raster
from hillshade import hillshade


def load_subtree_size_from_dfs(filename):
    a, b, c, d = raster.load(filename, bands=4)
    e = np.ascontiguousarray(np.transpose((a, b), (1, 2, 0))).view(np.uint64)
    f = np.ascontiguousarray(np.transpose((c, d), (1, 2, 0))).view(np.uint64)
    result = 1 + (f - e) // 2
    return result.reshape(a.shape)


DX = [0]*129
DY = [0]*129
DX[1] = DX[2] = DX[128] = 1  # east
DX[8] = DX[16] = DX[32] = -1  # west
DY[2] = DY[4] = DY[8] = 1  # south
DY[32] = DY[64] = DY[128] = -1  # north


def extract(subtree_size, dirs, y1, y2, x1, x2):
    subtree_size = subtree_size[y1-1:y2+1, x1-1:x2+1]
    dirs = dirs[y1-1:y2+1, x1-1:x2+1]
    n, m = subtree_size.shape
    for i in range(n):
        for j in range(m) if i in (0, n-1) else (0, m-1):
            dir = dirs[i, j]
            if dir in (0, 255):
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            if 0 < pi < n-1 and 0 < pj < m-1:
                assert subtree_size[pi, pj] > subtree_size[i, j]
                subtree_size[pi, pj] -= subtree_size[i, j]
    for i in range(1, n-1):
        for j in range(1, m-1) if i in (1, n-2) else (1, m-2):
            dir = dirs[i, j]
            if dir in (0, 255):
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            if (pi in (0, n-1)) or (pj in (0, m-1)):
                dirs[i, j] = 255  # Set dir to nodata
    return subtree_size[1:-1, 1:-1], dirs[1:-1, 1:-1]


def find_highlight(dirs, cx1, cx2, cy1, cy2):
    highlight = np.zeros(dirs.shape, np.uint8)
    highlight[cy1:cy2, cx1:cx2] = 1
    for i in range(cy1, cy2):
        for j in range(cx1, cx2) if i in (cy1, cy2-1) else (cx1, cx2-1):
            dir = dirs[i, j]
            if dir in (0, 255):
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            while not ((cy1 <= pi < cy2 and cx1 <= pj < cx2) or highlight[pi, pj]):
                highlight[pi, pj] = 1
                dir = dirs[pi, pj]
                if dir in (0, 255):
                    break
                pi += DY[dir]
                pj += DX[dir]
    return highlight


highlight_color = ['darkblue', 'blue']
highlight_pen = ['normal', 'heavier']


def main():
    subtree_size = load_subtree_size_from_dfs('mandoe-1m3-dfs.tif')
    dirs = raster.load('mandoe-1m3-dirs.tif')
    n, m = dirs.shape
    y1 = n//2-100
    y2 = n//2+100 - 139
    x1 = m//2-100 + 89
    x2 = m//2+100 - 50
    subtree_size, dirs = extract(subtree_size, dirs, y1, y2, x1, x2)
    hs = hillshade(raster.load('mandoe-1m3.tif')[y1-1:y2+1, x1-1:x2+1],
                   light_angle=45, light_azimuth=315, z_factor=20)
    data = np.repeat(hs.astype(np.uint8), 3).tobytes()
    data_compress = zlib.compress(data)
    length = len(data_compress)
    bitmap = (
        '<bitmap id="1" width="{width}" height="{height}" length="{length}" ' +
        'ColorSpace="DeviceRGB" Filter="FlateDecode" BitsPerComponent="8" ' +
        'encoding="base64">\n{data}\n</bitmap>\n'
    ).format(width=hs.shape[1], height=hs.shape[0], length=length,
             data=base64.b64encode(data_compress).decode('ascii'))
    image = '<image rect="-0.5 {y} {x} 0.5" bitmap="1"/>'.format(
        x=hs.shape[1]-0.5, y=-(hs.shape[0]-0.5))

    n, m = dirs.shape
    cx1 = m//3
    cx2 = m-cx1
    cy1 = n//3
    cy2 = n-cy1
    child_weight = np.zeros(dirs.shape, np.uint64)
    child_dir = np.zeros(dirs.shape, np.uint8)
    highlight = find_highlight(dirs, cx1, cx2, cy1, cy2)
    child_highlight = np.zeros_like(highlight)
    for i, row in raster.iterprogress(enumerate(dirs), n=len(dirs)):
        for j, dir in enumerate(row):
            if dir in (0, 255):
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            hi = highlight[i, j]
            assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0]), (i, j, dir, pi, pj)
            phi = child_highlight[pi, pj]
            if (phi, child_weight[pi, pj]) < (hi, subtree_size[i, j]):
                child_dir[pi, pj] = dir
                child_weight[pi, pj] = subtree_size[i, j]
                child_highlight[pi, pj] = hi
    print('<ipeselection pos="0 0">')
    print(bitmap)
    print('<group matrix="2 0 0 2 0 0">')
    print(image)
    for i, row in raster.iterprogress(enumerate(dirs), n=len(dirs)):
        for j, dir in enumerate(row):
            if dir == 0:
                continue
            if dir == 255:
                continue
            if subtree_size[i, j] > 1:
                continue
            pi = i + DY[dir]
            pj = j + DX[dir]
            assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0])
            print('<path stroke="%s" pen="%s">' %
                  (highlight_color[highlight[i, j]],
                   highlight_pen[highlight[i, j]]))
            print('%s %s m' % (j, -i))
            while True:
                assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0])
                print('%s %s l' % (pj, -pi))
                if child_dir[pi, pj] != dir:
                    break
                dir = dirs[pi, pj]
                if dir in (0, 255):
                    break
                pi += DY[dir]
                pj += DX[dir]
            print('</path>')
    for i, row in raster.iterprogress(enumerate(dirs), n=len(dirs)):
        for j, dir in enumerate(row):
            if dir == 0:
                print('<use name="mark/disk(sx)" pos="%s %s" ' % (j, -i) +
                      'size="small" stroke="blue"/>')
    print('<path stroke="brown" pen="ultrafat">')
    print('{x1} {y1} m\n{x2} {y1} l\n{x2} {y2} l\n{x1} {y2} l\nh'.format(
        x1=cx1, x2=cx2-1, y1=-cy1, y2=-(cy2-1)))
    print('{x1} {y1} m\n{x2} {y1} l\n{x2} {y2} l\n{x1} {y2} l\nh'.format(
        x1=0, x2=m-1, y1=0, y2=-(n-1)))
    print('</path>')
    print('</group>')
    print('</ipeselection>')


if __name__ == '__main__':
    main()
