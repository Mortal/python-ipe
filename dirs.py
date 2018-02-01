import os
import re
import sys
import zlib
import shlex
import base64
import argparse
import datetime
import tempfile
import textwrap
import itertools
import contextlib
import subprocess
import numpy as np
sys.path.append('/home/rav/pygdal')
import raster  # noqa
from hillshade import hillshade  # noqa


try:
    from griddfs import mark_downstream, mark_upstream
except ImportError:
    print("Could not import griddfs.")
    print("Consider running pip install --user https://github.com/Mortal/griddfs/releases/download/v0.2.0/griddfs-0.2.0-py2.py3-none-manylinux1_x86_64.whl")


PROG_NAME = 'dirs.py'


def load_subtree_size_from_dfs(filename, return_dfs=False, **kwargs):
    a, b, c, d = raster.load(filename, bands=4, **kwargs)
    e = np.ascontiguousarray(np.transpose((a, b), (1, 2, 0))).view(np.uint64)[:, :, 0]
    f = np.ascontiguousarray(np.transpose((c, d), (1, 2, 0))).view(np.uint64)[:, :, 0]
    result = 1 + (f - e) // 2
    result = result.reshape(a.shape)
    if not return_dfs:
        return result
    return result, (e, f)


DIR_E = (0, 1)
DIR_SE = (1, 1)
DIR_S = (1, 0)
DIR_SW = (1, -1)
DIR_W = (0, -1)
DIR_NW = (-1, -1)
DIR_N = (-1, 0)
DIR_NE = (-1, 1)
DIR_NAMES = [
    '\N{RIGHTWARDS ARROW}',
    '\N{SOUTH EAST ARROW}',
    '\N{DOWNWARDS ARROW}',
    '\N{SOUTH WEST ARROW}',
    '\N{LEFTWARDS ARROW}',
    '\N{NORTH WEST ARROW}',
    '\N{UPWARDS ARROW}',
    '\N{NORTH EAST ARROW}',
]
DIRS = [
    DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW, DIR_N, DIR_NE,
]


def print_dirs(dirs):
    names = ['\N{MIDDLE DOT}'] + DIR_NAMES
    print('\n'.join(''.join(names[int(v).bit_length()] for v in row)
                    for row in dirs))
    print('-'*dirs.shape[1])


def dirs_from_dfs(dfs):
    discover, finish = dfs
    assert discover.shape == finish.shape
    assert discover.ndim == 2, discover.shape
    subtree_size = (1 + (finish - discover) // 2).astype(np.int64)
    discover_pad = np.pad(discover, 1, 'constant')
    finish_pad = np.pad(finish, 1, 'constant')
    subtree_size = np.pad(subtree_size, 1, 'constant')
    maxint = np.iinfo(subtree_size.dtype).max

    contained = []
    slices = [slice(0, -2), slice(1, -1), slice(2, None)]
    for i, j in DIRS:
        dir_slice = (slices[i+1], slices[j+1])
        # Compare discover times to the (d,f)-interval of the (i,j)-neighbor
        contained.append(np.where(
            (discover_pad[dir_slice] < discover) &
            (discover < finish_pad[dir_slice]),
            subtree_size[dir_slice],
            maxint))
    contained = np.asarray(contained)
    assert contained.dtype == subtree_size.dtype
    assert contained.shape == (8,) + discover.shape, contained.shape
    # Compute direction index for each cell
    direction_index = np.argmin(contained, axis=0)
    # A cell is a non-sink if it is contained
    # in the neighbor pointed to by its direction index
    has_dir = (np.choose(direction_index, contained) != maxint)

    res = np.zeros(discover.shape, np.uint8)
    # Set flow directions for non-sinks to a non-zero value
    res[has_dir] = 1 << direction_index[has_dir]
    return res


def neighbor(pos, dir):
    i, j = pos
    if dir in (0, 255):
        return None
    dx, dy = DIRS[int(dir).bit_length()-1]
    i += dx
    j += dy
    return i, j


def extract(subtree_size, dirs):
    n, m = subtree_size.shape
    for i in range(n):
        for j in range(m) if i in (0, n-1) else (0, m-1):
            dir = dirs[i, j]
            if dir in (0, 255):
                continue
            pi, pj = neighbor((i, j), dir)
            if 0 < pi < n-1 and 0 < pj < m-1:
                assert subtree_size[pi, pj] > subtree_size[i, j]
                subtree_size[pi, pj] -= subtree_size[i, j]
    for i in range(1, n-1):
        for j in range(1, m-1) if i in (1, n-2) else (1, m-2):
            dir = dirs[i, j]
            if dir in (0, 255):
                continue
            pi, pj = neighbor((i, j), dir)
            if (pi in (0, n-1)) or (pj in (0, m-1)):
                dirs[i, j] = 255  # Set dir to nodata
    # Make copy to ensure contiguous result
    return np.array(subtree_size[1:-1, 1:-1]), np.array(dirs[1:-1, 1:-1])


MARK_FROM_INNER = 2**0
MARK_FROM_OUTER = 2**1
MARK_TO_OUTER = 2**2
MARK_TO_INNER = 2**3

MARK_OUT = MARK_FROM_INNER | MARK_TO_OUTER
MARK_IN = MARK_FROM_OUTER | MARK_TO_INNER


def find_highlight(dirs, cx1, cx2, cy1, cy2):
    n, m = dirs.shape
    marks = mark_downstream(dirs, (cy1, cx1, cx2-cx1, cy2-cy1), mark=MARK_FROM_INNER)
    mark_downstream(dirs, (0, 0, m, 1), marks, mark=MARK_FROM_OUTER)
    mark_downstream(dirs, (1, 0, 1, n-2), marks, mark=MARK_FROM_OUTER)
    mark_downstream(dirs, (1, m-1, 1, n-2), marks, mark=MARK_FROM_OUTER)
    mark_downstream(dirs, (n-1, 0, m, 1), marks, mark=MARK_FROM_OUTER)
    mark_upstream(dirs, (0, 0, m, 1), marks, mark=MARK_TO_OUTER)
    mark_upstream(dirs, (1, 0, 1, n-2), marks, mark=MARK_TO_OUTER)
    mark_upstream(dirs, (1, m-1, 1, n-2), marks, mark=MARK_TO_OUTER)
    mark_upstream(dirs, (n-1, 0, m, 1), marks, mark=MARK_TO_OUTER)
    mark_upstream(dirs, (cy1, cx1, cx2-cx1, cy2-cy1), marks, mark=MARK_TO_INNER)
    return marks


def highlight_style(mark):
    if mark & MARK_OUT == MARK_OUT:
        return 'blue', 'heavier'
    if mark & MARK_IN == MARK_IN:
        return 'purple', 'heavier'
    return 'darkblue', 'normal'


def parse_rect_arg(s):
    if s.startswith(('+', '-')):
        cx, cy, w, h = s.split(',')
        cx, cy = map(float, (cx, cy))
        w, h = map(int, (w, h))
        if min((w, h)) < 0:
            raise ValueError('Rectangle lengths must be non-negative')
        return 'coords', cx, cy, w, h
    else:
        x, y, w, h = map(int, s.split(','))
        if min((x, y, w, h)) < 0:
            raise ValueError('Rectangle values must be non-negative')
        return 'grid', x, y, w, h


parser = argparse.ArgumentParser()
parser.add_argument('-z', '--z-factor', type=float, default=10,
                    help='z-factor used in hillshading')
parser.add_argument('--input-elev',
                    help='Elevation model to use for hillshaded background')
parser.add_argument('-r', '--rect', metavar='X,Y,W,H', type=parse_rect_arg,
                    help='rectangle to render (top-left corner + size; ' +
                    'default: all). If the string starts with "+" or "-", ' +
                    'then X and Y are the rectangle center in coordinates; ' +
                    'otherwise, X and Y are top-left column/row offsets.')
parser.add_argument('--input-dfs', required=True,
                    help='DFS numbering raster')
parser.add_argument('--input-dirs',
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
        elif output.endswith('.pdf'):
            cwd = os.getcwd()
            quoted_argv = ' '.join(shlex.quote(arg) for arg in sys.argv)
            fp = stack.enter_context(tempfile.NamedTemporaryFile('w', suffix='.ipe'))
        else:
            cwd = os.getcwd()
            quoted_argv = ' '.join(shlex.quote(arg) for arg in sys.argv)
            fp = stack.enter_context(open(output, 'w'))
        with IpeDoc(fp) as ipedoc:
            args['ipedoc'] = ipedoc
            yield args
        if output and output.endswith('.pdf'):
            fp.flush()
            subprocess.check_call(('ipetoipe', '-pdf', fp.name, output))

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
    def read_ipestyle(self, name='basic'):
        filename = '/usr/share/ipe/%s/styles/%s.isy' % (
            '.'.join(map(str, self.version)), name)
        with open(filename) as fp:
            xml_ver = fp.readline()
            doctype = fp.readline()
            assert xml_ver == '<?xml version="1.0"?>\n'
            assert doctype == '<!DOCTYPE ipestyle SYSTEM "ipe.dtd">\n'
            return fp.read().rstrip()

    @classmethod
    def find_version(cls):
        mo = max((re.match(r'^(\d+)\.(\d+)\.(\d+)$', v)
                  for v in os.listdir('/usr/share/ipe')),
                 key=lambda mo: (mo is not None, mo and mo.group(0)))
        return tuple(map(int, mo.group(1, 2, 3)))

    def __init__(self, output_fp):
        self.output_fp = output_fp
        self.bitmaps = []
        self.version = self.find_version()

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.output_fp)

    def __enter__(self):
        self.print('<?xml version="1.0"?>')
        self.print('<!DOCTYPE ipe SYSTEM "ipe.dtd">')
        version = '%d%02d%02d' % self.version
        self.print('<ipe version="%s" creator="%s">' % (version, PROG_NAME))
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


def main(ipedoc, input_dfs, input_dirs=None, input_elev=None,
         rect=None, light_angle=45, light_azimuth=315, z_factor=20):
    if rect is None:
        read_args = {}
    else:
        kind, x1, y1, width, height = rect
        if kind == 'coords':
            ds = raster.gdal.Open(input_dfs)
            x0, dx, _1, y0, _2, dy = ds.GetGeoTransform()
            assert _1 == _2 == 0
            x1 = int((x1-x0) / dx - width/2)
            y1 = int((y1-y0) / dy - height/2)
            if not 0 <= x1 < ds.RasterXSize - width:
                raise SystemExit('x-coordinates outside raster ' +
                                 str((x1, ds.RasterXSize, width, ds.GetGeoTransform())))
            if not 0 <= y1 < ds.RasterYSize - height:
                raise SystemExit('y-coordinates outside raster ' +
                                 str((y1, ds.RasterYSize, height, ds.GetGeoTransform())))
            # TODO support computing on the boundary of the raster
            if not 0 < x1 < ds.RasterXSize-width - 1:
                raise SystemExit('Rectangle must be strictly within raster')
            if not 0 < y1 < ds.RasterYSize-height - 1:
                raise SystemExit('Rectangle must be strictly within raster')
            del ds
        else:
            assert kind == 'grid'
        read_args = dict(offset=(x1-1, y1-1), size=(width+2, height+2))

    if input_dirs is None:
        subtree_size, dfs = load_subtree_size_from_dfs(
            input_dfs, **read_args, return_dfs=True)
        dirs = dirs_from_dfs(dfs)
    else:
        # subtree_size, dfs = load_subtree_size_from_dfs(
        #     input_dfs, **read_args, return_dfs=True)
        # dirs = raster.load(input_dirs, **read_args)
        # dirs_test = dirs_from_dfs(dfs)
        # assert np.all(dirs[1:-1, 1:-1] == dirs_test[1:-1, 1:-1])
        subtree_size = load_subtree_size_from_dfs(
            input_dfs, **read_args)
        dirs = raster.load(input_dirs, **read_args)

    subtree_size, dirs = extract(subtree_size, dirs)
    n, m = dirs.shape
    scale = 2
    if input_elev:
        hs = hillshade(raster.load(input_elev, **read_args),
                       light_angle, light_azimuth, z_factor)
        bitmap_id = ipedoc.add_bitmap(hs.astype(np.uint8))
        # rect is lower left xy, upper right xy
        image = '<image rect="0 0 {m} {n}" bitmap="{id}" matrix="{s} 0 0 {s} 0 0"/>'.format(
            m=hs.shape[1], n=hs.shape[0], id=bitmap_id, s=scale)
    else:
        image = None

    cx1 = m//3
    cx2 = m-cx1
    cy1 = n//3
    cy2 = n-cy1
    child_weight = np.zeros(dirs.shape, np.uint64)
    child_dir = np.zeros(dirs.shape, np.uint8)
    highlight = find_highlight(dirs, cx1, cx2, cy1, cy2)
    for i, row in enumerate(dirs):
        for j, dir in enumerate(row):
            if dir in (0, 255):
                continue
            pi, pj = neighbor((i, j), dir)
            assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0]), (i, j, dir, pi, pj)
            if child_weight[pi, pj] < subtree_size[i, j]:
                child_dir[pi, pj] = dir
                child_weight[pi, pj] = subtree_size[i, j]
    with ipedoc.page() as page:
        # (i,j)->(2j,2n-2i)
        if image:
            page.print(image)
        with page.group(0, -scale, scale, 0, scale*0.5, scale*(n-0.5)) as group:
            output_dirs(group, image, dirs, highlight, subtree_size, child_dir,
                        cx1, cx2, cy1, cy2)


def follow_selected_path(dirs, child_dir, i, j):
    dir = dirs[i, j]
    assert dir not in (0, 255)
    n = neighbor((i, j), dir)
    while n is not None:
        pi, pj = n
        assert 0 <= pi < len(dirs) and 0 <= pj < len(dirs[0])
        yield (pi, pj)
        if child_dir[pi, pj] != dir:
            break
        dir = dirs[pi, pj]
        n = neighbor((pi, pj), dir)


def output_dirs(group, image, dirs, highlight, subtree_size, child_dir,
                cx1, cx2, cy1, cy2):
    n, m = dirs.shape
    for i, row in enumerate(dirs):
        for j, dir in enumerate(row):
            if dir in (0, 255) or subtree_size[i, j] > 1:
                continue
            hi = highlight[i, j]
            path = follow_selected_path(dirs, child_dir, i, j)
            cur_path = [(i, j, 'm')]
            parts = itertools.groupby(
                path, key=lambda pos: highlight_style(highlight[pos]))
            for (stroke, pen), part in parts:
                for ii, jj in part:
                    cur_path.append((ii, jj, 'l'))
                group.path(cur_path, stroke=stroke, pen=pen)
                cur_path = [(ii, jj, 'm')]
    for i, row in enumerate(dirs):
        for j, dir in enumerate(row):
            if dir == 0:
                group.mark(i, j, 'disk', size='small', stroke='blue')
    group.path(
        [(cy1, cx1, 'm'),
         (cy1, cx2-1, 'l'),
         (cy2-1, cx2-1, 'l'),
         (cy2-1, cx1, 'l'),
         ('h',),
         (0, 0, 'm'),
         (0, m-1, 'l'),
         (n-1, m-1, 'l'),
         (n-1, 0, 'l'),
         ('h',)],
        stroke='brown', pen='ultrafat')


if __name__ == '__main__':
    with parse_args() as args:
        main(**args)
