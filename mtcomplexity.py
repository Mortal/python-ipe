import struct
import argparse
import collections
import numpy as np

from numba import jit

from tpie_stream import FORMATS


def parse_header(b):
    header_keys = (
        'magic version item_size block_size user_data_size ' +
        'max_user_data_size size flags last_block_read_offset').split()
    header_fmt = struct.Struct(len(header_keys) * 'L')
    header_dict = collections.OrderedDict(
        zip(header_keys, header_fmt.unpack(b[:header_fmt.size].tostring())))
    return header_dict


@jit(nopython=True)
def foo(items, basin, parent, rec, fwd):
    nblocks, blockitems = basin.shape
    roots = 0
    for k in range(items):
        i, j = k // blockitems, k % blockitems
        if parent[i, j] == 0:
            roots += 1
    return roots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', choices=FORMATS.keys(), required=True)
    parser.add_argument('filename')
    args = parser.parse_args()
    keys, fmt = FORMATS[args.format]
    fmt_nopad = fmt.rstrip('x')
    padding = len(fmt) - len(fmt_nopad)
    to_dtype = dict(l='i8', d='f8', I='u4', f='f4', i='i4')
    dt = list(zip(keys, (to_dtype[f] for f in fmt)))
    if padding:
        dt.append(('_padding', 'V%s' % padding))
    dt = np.dtype(dt)

    fp = np.memmap(args.filename, mode='r')
    header = parse_header(fp)
    data = fp[4096:]
    BS = 2**21
    nblocks, remaining = divmod(len(data), BS)
    assert remaining == 0
    blockitems, blockpadding = divmod(BS, dt.itemsize)
    items = np.ndarray((nblocks, blockitems), buffer=data.data, dtype=dt,
                       strides=(BS, dt.itemsize))
    print(foo(header['size'], items['basin'], items['parent'],
              items['rec'], items['fwd']))


if __name__ == "__main__":
    main()
