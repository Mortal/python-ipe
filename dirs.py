import os
import sys
import glob
import zlib
import shlex
import base64
import argparse
import datetime
import textwrap
import contextlib
import numpy as np
sys.path.append('/home/rav/pygdal')
import raster  # noqa
from hillshade import hillshade  # noqa


PROG_NAME = 'dirs.py'


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


def parse_rect_arg(s):
    x, y, w, h = map(int, s.split(','))
    if min((x, y, w, h)) < 0:
        raise ValueError('Rectangle values must be non-negative')
    return x, y, w, h


parser = argparse.ArgumentParser()
parser.add_argument('-z', '--z-factor', type=float, default=10,
                    help='z-factor used in hillshading')
parser.add_argument('--input-elev',
                    help='Elevation model to use for hillshaded background')
parser.add_argument('-r', '--rect', metavar='X,Y,W,H', type=parse_rect_arg,
                    help='rectangle to render (top-left corner + size; ' +
                    'default: all)')
parser.add_argument('--input-dfs', required=True,
                    help='DFS numbering raster')
parser.add_argument('--input-dirs', required=True,
                    help='Flow directions raster')
parser.add_argument('-o', '--output',
                    help='Ipe selection output file (default: stdout)')


@contextlib.contextmanager
def parse_args():
    args = vars(parser.parse_args())
    output = args.pop('output', None)
    with contextlib.ExitStack() as stack:
        if output is None:
            fp = sys.stdout
        else:
            cwd = os.getcwd()
            quoted_argv = ' '.join(shlex.quote(arg) for arg in sys.argv)
            fp = stack.enter_context(open(output, 'w'))
        args['ipedoc'] = stack.enter_context(IpeDoc(fp))
        yield args

    if output is not None:
        # Only write rerun script if output is a regular file.
        # This handles two cases: when output to special file (e.g. pipe),
        # and when no output file was created because of some error.
        if os.path.isfile(output):
            with open(output + '-rerun.sh', 'w') as fp:
                fp.write(textwrap.dedent('''
                #!/bin/bash
                cd {cwd}
                python3 {args}
                ''').lstrip().format(cwd=cwd, args=quoted_argv))


class IpeDoc:
    @classmethod
    def read_ipestyle(cls, name='basic'):
        pattern = '/usr/share/ipe/*/styles/%s.isy' % name
        filenames = glob.glob(pattern)
        if not filenames:
            raise FileNotFoundError(pattern)
        with open(max(filenames)) as fp:
            xml_ver = fp.readline()
            doctype = fp.readline()
            assert xml_ver == '<?xml version="1.0"?>\n'
            assert doctype == '<!DOCTYPE ipestyle SYSTEM "ipe.dtd">\n'
            return fp.read().rstrip()

    def __init__(self, output_fp):
        self.output_fp = output_fp
        self.bitmaps = []

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.output_fp)

    def __enter__(self):
        self.print('<?xml version="1.0"?>')
        self.print('<!DOCTYPE ipe SYSTEM "ipe.dtd">')
        self.print('<ipe version="70206" creator="%s">' % PROG_NAME)
        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.print('<info created="D:%s" modified="D:%s"/>' % (t, t))
        return self

    def __exit__(self, et, ev, eb):
        self.print('</ipe>')

    def add_bitmap(self, data):
        height, width = data.shape
        assert data.dtype == np.uint8
        bitmap_id = len(self.bitmaps) + 1
        data = np.repeat(data, 3).tobytes()
        data_compress = zlib.compress(data)
        length = len(data_compress)
        bitmap = (
            '<bitmap id="{id}" ' +
            'width="{width}" height="{height}" length="{length}" ' +
            'ColorSpace="DeviceRGB" Filter="FlateDecode" ' +
            'BitsPerComponent="8" encoding="base64">\n{data}\n</bitmap>'
        ).format(width=width, height=height, length=length, id=bitmap_id,
                 data=base64.b64encode(data_compress).decode('ascii'))
        self.bitmaps.append(bitmap)
        return bitmap_id

    @contextlib.contextmanager
    def page(self):
        for bitmap in self.bitmaps:
            self.print(bitmap)
        self.bitmaps = None
        self.print(self.read_ipestyle())
        self.print('<page>')
        try:
            yield self
        finally:
            self.print('</page>')

    @contextlib.contextmanager
    def group(self, *matrix):
        assert len(matrix) == 6
        assert all(isinstance(v, (int, float)) for v in matrix), matrix
        self.print('<group matrix="%s">' % ' '.join('%g' % v for v in matrix))
        try:
            yield self
        finally:
            self.print('</group>')

    def path(self, path, **attrs):
        self.print('<path%s>' %
                   ''.join(' %s="%s"' % kv for kv in attrs.items()))
        for vals in path:
            self.print(' '.join(map(str, vals)))
        self.print('</path>')

    def mark(self, x, y, name, **attrs):
        self.print('<use name="mark/%s(sx)" pos="%s %s"' % (name, x, y) +
                   '%s/>' % ''.join(' %s="%s"' % kv for kv in attrs.items()))


def main(ipedoc, input_dfs, input_dirs, input_elev=None,
         rect=None, light_angle=45, light_azimuth=315, z_factor=20):
    subtree_size = load_subtree_size_from_dfs(input_dfs)
    dirs = raster.load(input_dirs)
    n, m = dirs.shape
    if rect is None:
        x1, y1 = 1, 1
        x2, y2 = m-1, n-1
    else:
        x1, y1, width, height = rect
        x2 = x1 + width
        y2 = y1 + height
    subtree_size, dirs = extract(subtree_size, dirs, y1, y2, x1, x2)
    if input_elev:
        hs = hillshade(raster.load(input_elev)[y1-1:y2+1, x1-1:x2+1],
                       light_angle, light_azimuth, z_factor)
        bitmap_id = ipedoc.add_bitmap(hs.astype(np.uint8))
        image = '<image rect="-0.5 {y} {x} 0.5" bitmap="{id}"/>'.format(
            x=hs.shape[1]-0.5, y=-(hs.shape[0]-0.5), id=bitmap_id)
    else:
        image = None

    n, m = dirs.shape
    cx1 = m//3
    cx2 = m-cx1
    cy1 = n//3
    cy2 = n-cy1
    child_weight = np.zeros(dirs.shape, np.uint64)
    child_dir = np.zeros(dirs.shape, np.uint8)
    highlight = find_highlight(dirs, cx1, cx2, cy1, cy2)
    child_highlight = np.zeros_like(highlight)
    for i, row in enumerate(dirs):
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
    with ipedoc.page() as page:
        with page.group(2, 0, 0, 2, 0, 0) as group:
            if image:
                group.print(image)
            output_dirs(group, image, dirs, highlight, subtree_size, child_dir,
                        cx1, cx2, cy1, cy2)


def output_dirs(group, image, dirs, highlight, subtree_size, child_dir,
                cx1, cx2, cy1, cy2):
    n, m = dirs.shape
    for i, row in enumerate(dirs):
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
            path = [(j, -i, 'm')]
            while True:
                assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0])
                path.append((pj, -pi, 'l'))
                if child_dir[pi, pj] != dir:
                    break
                dir = dirs[pi, pj]
                if dir in (0, 255):
                    break
                pi += DY[dir]
                pj += DX[dir]
            group.path(path, stroke=highlight_color[highlight[i, j]],
                       pen=highlight_pen[highlight[i, j]])
    for i, row in enumerate(dirs):
        for j, dir in enumerate(row):
            if dir == 0:
                group.mark(j, -i, 'disk', size='small', stroke='blue')
    group.path(
        [(cx1, -cy1, 'm'),
         (cx2-1, -cy1, 'l'),
         (cx2-1, -(cy2-1), 'l'),
         (cx1, -(cy2-1), 'l'),
         ('h',),
         (0, 0, 'm'),
         (m-1, 0, 'l'),
         (m-1, -(n-1), 'l'),
         (0, -(n-1), 'l'),
         ('h',)],
        stroke='brown', pen='ultrafat')


if __name__ == '__main__':
    with parse_args() as args:
        main(**args)
