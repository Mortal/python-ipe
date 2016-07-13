import time
import struct
import argparse
import collections
import numpy as np

from numba import jit, njit

from tpie_stream import parse_header


FORMATS = {
    'merge-tree': (
        'basin V A rec fwd parent z'.split(),
        'lddIIlfxxxx',
    ),
    'merge-tree-2': (
        'basin V A rec fwd parent i j z r'.split(),
        'lddIIliifI',
    ),
}


@njit
def getflat(arr, idx):
    n, m = arr.shape
    i, j = idx // m, idx % m
    return arr[i, j]


@njit
def compute_subtree_size(items, parent):
    subtree_size = np.zeros(items, dtype=np.uint32)
    for i in range(items):
        subtree_size[i] += 1
        if getflat(parent, i) != 0:
            subtree_size[getflat(parent, i) - 1] += subtree_size[i]
    return subtree_size


@njit
def count_leaves(fwd):
    nblocks, blockitems = fwd.shape
    k = 0
    nleaves = 0
    for i in range(nblocks):
        for j in range(blockitems):
            k += 1
            if fwd[i, j] == k:
                nleaves += 1
    return nleaves


@njit
def partition(x, v):
    left_size = 0
    for i in range(len(x)):
        if x[i] < v:
            x[left_size], x[i] = x[i], x[left_size]
            left_size += 1
    return left_size


@njit('(int64, Array(int64, 2, "A", readonly=True), \
        Array(uint32, 2, "A", readonly=True))')
def count_signals(items, parent, fwd):
    nblocks, blockitems = parent.shape
    roots = 0

    subtree_size = compute_subtree_size(items, parent)

    signals = np.zeros(items, dtype=np.uint32)
    stack_size = np.zeros(items, dtype=np.uint32)
    stack_owner = np.zeros(items, dtype=np.uint32)
    tos = 0

    signal_count = 0

    for k in range(items - 1, -1, -1):
        i, j = k // blockitems, k % blockitems
        # print(k, parent[i, j])

        if parent[i, j] > 0:
            while tos > 0 and stack_owner[tos] != parent[i, j]:
                # The top of stack was from a node that has no more
                # children, so it should be empty.
                if stack_size[tos-1] != stack_size[tos]:
                    raise Exception("Unhandled signals on top of stack")
                tos -= 1
            # Our parent has forwarded a list to us on the stack.
            if tos == 0:
                raise Exception("Parent did not forward a list of signals")

        # Push empty stack
        tos += 1
        stack_size[tos] = stack_size[tos-1]
        stack_owner[tos] = k+1

        if parent[i, j] == 0:
            roots += 1
        else:
            # Partition parent list into ours and not ours
            min_leaf = k - subtree_size[k] + 1
            not_ours = partition(
                signals[stack_size[tos-2]:stack_size[tos]], min_leaf)
            stack_size[tos-1] = stack_size[tos-2] + not_ours

        signal_count += stack_size[tos] - stack_size[tos-1]
        if fwd[i, j] == k+1:
            if subtree_size[k] != 1:
                raise Exception("Leaf, but subtree size is not 1")
            # Leaf
            for s in signals[stack_size[tos-1]:stack_size[tos]]:
                if s != k:
                    raise Exception("Signal for leaf not to leaf")
            tos -= 1
        else:
            if subtree_size[k] == 1:
                raise Exception("Non-leaf, but subtree size is 1")
            signals[stack_size[tos]] = fwd[i, j] - 1
            stack_size[tos] += 1

    return signal_count


@njit
def count_nonroots(parent):
    nblocks, blockitems = parent.shape
    nparents = 0
    for i in range(nblocks):
        for j in range(blockitems):
            if parent[i, j] != 0:
                nparents += 1
    return nparents


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

    header_area_size = 4096

    with open(args.filename, 'r+b') as fp:
        header_area = fp.read(header_area_size)
        header = parse_header(header_area)
        basin_count = header['size']
        block_size = header['block_size']
        assert block_size == 2**21
        assert header['item_size'] == dt.itemsize
        assert header['max_user_data_size'] <= 8
        block_items, block_padding = divmod(block_size, dt.itemsize)
        nblocks, last_block_size = divmod(basin_count, block_items)
        if last_block_size > 0:
            nblocks += 1
        expected_size = nblocks * block_size
        fp.seek(0, 2)  # seek to end
        actual_size = fp.tell()
        if actual_size < expected_size:
            padding = expected_size - actual_size
            print("Extending %s with %s zero bytes" %
                  (args.filename, padding))
            assert padding < block_size
            fp.write(b'\x00' * padding)

    fp = np.memmap(args.filename, mode='r')
    data = fp[header_area_size:]
    merge_tree = np.ndarray(
        (nblocks, block_items), buffer=data.data, dtype=dt,
        strides=(block_size, dt.itemsize))
    basin_count = header['size']
    print("Number of basins: %s" % basin_count)
    parent = merge_tree['parent']
    forwarder = merge_tree['fwd']
    sinks = count_leaves(forwarder)
    print("Number of sinks: %s" % sinks)
    saddles = basin_count - sinks
    print("Number of internal nodes: %s" % saddles)
    trickle_path_edges = count_signals(basin_count, parent, forwarder)
    print("Number of trickle path edges: %s" % trickle_path_edges)
    print("On average %.2f edges per saddle" % (trickle_path_edges / saddles))


if __name__ == "__main__":
    main()
