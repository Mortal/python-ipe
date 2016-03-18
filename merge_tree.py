import sys
import json
import struct
import argparse
import collections


def iterblocks(filename):
    with open(filename, 'rb') as fp:
        header = fp.read(4096)
        header_keys = (
            'magic version item_size block_size user_data_size ' +
            'max_user_data_size size flags last_block_read_offset').split()
        header_fmt = struct.Struct(len(header_keys) * 'L')
        print(json.dumps(collections.OrderedDict(zip(header_keys, header_fmt.unpack(header[:header_fmt.size])))))
        while True:
            block = fp.read(2048*1024)
            if not block:
                break
            yield block


def iteritems(filename, item_size):
    print(item_size)
    for block in iterblocks(filename):
        for i in range(0, len(block) - item_size + 1, item_size):
            j = i + item_size
            yield block[i:j]


def iterstructs(filename, fmt):
    s = struct.Struct(fmt)
    for item in iteritems(filename, s.size):
        yield s.unpack(item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    keys = 'basin V A rec fwd parent z'.split()
    fmt = 'lddIIlfxxxx'
    basin_format = '%7d'
    float_format = '%22s'
    display_format = {
        'basin': basin_format,
        'rec': basin_format,
        'fwd': basin_format,
        'parent': basin_format,
        'V': float_format,
        'z': float_format,
        'A': '%6.0f.0',
    }

    for v in iterstructs(args.filename, fmt):
        print("{%s}" %
              ', '.join('"%s": %s' % (k, display_format[k] % (a,))
                        for k, a in zip(keys, v)))


if __name__ == "__main__":
    main()
