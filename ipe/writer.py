import os
import re
import zlib
import base64
import datetime
import contextlib
import numpy as np


PROG_NAME = 'python-ipe'


class IpeStyleMixin:
    def read_ipestyle(self, name='basic'):
        filename = '/usr/share/ipe/%s/styles/%s.isy' % (
            '.'.join(map(str, self.ipe_version)), name)
        with open(filename) as fp:
            xml_ver = fp.readline()
            doctype = fp.readline()
            assert xml_ver == '<?xml version="1.0"?>\n'
            assert doctype == '<!DOCTYPE ipestyle SYSTEM "ipe.dtd">\n'
            return fp.read().rstrip()

    def _find_version():
        mo = max((re.match(r'^(\d+)\.(\d+)\.(\d+)$', v)
                  for v in os.listdir('/usr/share/ipe')),
                 key=lambda mo: (mo is not None, mo and mo.group(0)))
        return tuple(map(int, mo.group(1, 2, 3)))

    ipe_version = _find_version()

    def get_preamble(self, prog_name=PROG_NAME):
        version = '%d%02d%02d' % self.ipe_version
        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        return (
            '<?xml version="1.0"?>\n' +
            '<!DOCTYPE ipe SYSTEM "ipe.dtd">\n' +
            '<ipe version="%s" creator="%s">\n' % (version, prog_name) +
            '<info created="D:%s" modified="D:%s"/>\n' % (t, t)
        )

    def get_postamble(self):
        return '</ipe>\n'


class IpeDoc(IpeStyleMixin):
    def __init__(self, output_fp):
        super().__init__()
        self.output_fp = output_fp
        self.bitmaps = []

    def print(self, *args, **kwargs):
        print(*args, **kwargs, file=self.output_fp)

    def __enter__(self):
        self.print(self.get_preamble().rstrip('\n'))
        return self

    def __exit__(self, et, ev, eb):
        self.print(self.get_postamble().rstrip('\n'))

    def add_bitmap(self, data):
        assert data.dtype == np.uint8, data.dtype
        bitmap_id = len(self.bitmaps) + 1
        if data.ndim == 2:
            height, width = data.shape
            data = np.repeat(data, 3)
        elif data.ndim == 3:
            height, width, depth = data.shape
            assert depth == 3
        data = data.tobytes()
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

    def text(self, x, y, content, **attrs):
        attrs.setdefault('type', 'label')
        attrs['pos'] = '%s %s' % (x, y)
        self.print('<text%s>%s</text>' %
                   (''.join(' %s="%s"' % kv for kv in attrs.items()),
                    content))

    def image(self, bitmap_id, x1, y1, x2, y2):
        assert x1 < x2
        assert y1 < y2
        # rect is lower left xy, upper right xy
        self.print('<image rect="%s %s %s %s" bitmap="%s"/>' %
                   (x1, y1, x2, y2, bitmap_id))
