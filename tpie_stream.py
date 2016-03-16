import sys
import json
import struct
import argparse
import collections


FORMATS = {
    'merge-tree': (
        'basin V A rec fwd parent z'.split(),
        'lddIIlfxxxx',
    ),
    'dfs-merge-tree': (
        'basin parent V z rec1 rec2 split'.split(),
        'IIffIII',
    ),
}


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
    parser.add_argument('--format', choices=FORMATS.keys(), required=True)
    parser.add_argument('filename')
    args = parser.parse_args()
    keys, fmt = FORMATS[args.format]
    for v in iterstructs(args.filename, fmt):
        print(json.dumps(collections.OrderedDict(zip(keys, v)), sys.stdout))


if __name__ == "__main__":
    main()
