import sys
import json
import struct
import argparse
import collections


PY2 = sys.hexversion <= 0x03000000


if PY2:
    from itertools import izip as zip


FORMATS = {
    'merge-tree': (
        'basin V A rec fwd parent z'.split(),
        'lddIIlfxxxx',
    ),
    'merge-tree-2': (
        'basin V A rec fwd parent i j z r'.split(),
        'lddIIliifI',
    ),
    'dfs-merge-tree': (
        'basin parent V z rec1 rec2 split'.split(),
        'IIffIII',
    ),
}


def parse_header(b):
    header_keys = (
        'magic version item_size block_size user_data_size ' +
        'max_user_data_size size flags last_block_read_offset').split()
    header_fmt = struct.Struct(len(header_keys) * 'L')
    header_dict = collections.OrderedDict(
        zip(header_keys, header_fmt.unpack(b[:header_fmt.size])))
    return header_dict


def iterblocks(filename, item_size):
    with open(filename, 'rb') as fp:
        header = fp.read(4096)
        header_dict = parse_header(header)
        print(json.dumps(header_dict))
        if item_size != header_dict['item_size']:
            raise ValueError(
                "Item size mismatch: Header says %s, expected %s" %
                (header_dict['item_size'], item_size))
        while True:
            block = fp.read(2048*1024)
            if not block:
                break
            yield block


def iteritems(filename, item_size):
    print(item_size)
    for block in iterblocks(filename, item_size):
        for i in range(0, len(block) - item_size + 1, item_size):
            j = i + item_size
            yield block[i:j]


class float32(object):
    """
    >>> str(float32(1.659999966621399))
    '1.66'
    >>> float32(1.659999966621399)
    float32(1.66)
    """

    def __init__(self, v):
        self.v = v

    def __str__(self):
        # At least 6 (digits10) but not 9 (max_digits10)
        return '%.6g' % (self.v,)

    def __repr__(self):
        return 'float32(%s)' % (self,)

    def dumps(self):
        return str(self)


def unpack(fmt, s, item):
    r = []
    for f, o in zip(fmt, s.unpack(item)):
        if f == 'f':
            r.append(float32(o))
        else:
            r.append(o)
    return tuple(r)


def iterstructs(filename, fmt):
    s = struct.Struct(fmt)
    for item in iteritems(filename, s.size):
        yield unpack(fmt, s, item)


def dumps(v):
    try:
        return v.dumps()
    except AttributeError:
        return json.dumps(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=FORMATS.keys(), required=True)
    parser.add_argument('filename')
    args = parser.parse_args()
    keys, fmt = FORMATS[args.format]
    for v in iterstructs(args.filename, fmt):
        o = collections.OrderedDict(zip(keys, v))
        print('{%s}' % ', '.join('%s: %s' % (json.dumps(k), dumps(v)) for k, v in o.items()))


if __name__ == "__main__":
    main()
