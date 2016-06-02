import re
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--translate')
    parser.add_argument('depth1')
    parser.add_argument('depth2')
    parser.add_argument('subtrees')
    parser.add_argument('threshold', type=float)
    args = parser.parse_args()

    if args.translate is None:
        tr = None
    else:
        tr = []
        with open(args.translate) as fp:
            for line in fp:
                k, v = line.split()
                tr.append((int(k), int(v)))
        tr = dict(tr)

    depths = []
    with open(args.depth1) as f1, open(args.depth2) as f2:
        for l1, l2 in zip(f1, f2):
            o1, o2 = json.loads(l1), json.loads(l2)
            w1 = o1.get('watershed')
            w2 = o2.get('watershed')
            if w1 != w2:
                raise ValueError((w1, w2))
            r1 = o1['total_rain']
            r2 = o2['total_rain']
            if w1 is None:
                print("Difference in total rain: %s - %s = %s" %
                      (r1, r2, r1 - r2))
                continue
            a1 = o1['area']
            a2 = o2['area']
            if a1 != a2:
                raise ValueError((a1, a2))
            d1 = o1['max'] - o1['min']
            d2 = o2['max'] - o2['min']
            d = o1['max'] - o2['max']
            if abs(d1) > 0.0001:
                print("%s: File 1 has large difference: %s - %s = %s" %
                      (w1, o1['max'], o1['min'], d1),
                      file=sys.stderr)
            if abs(d2) > 0.0001:
                print("%s: File 2 has large difference: %s - %s = %s" %
                      (w1, o2['max'], o2['min'], d2),
                      file=sys.stderr)
            if w1+1 > len(depths):
                depths.extend([None] * (w1+1 - len(depths)))
            depths[w1] = (o1['max'], o2['max'], r1, r2, a1)
    with open(args.subtrees) as fp:
        pattern = (r'(\d+):(\d+) (full|partially_full) V=([-\d\.]+) ' +
                   r'basinVolume=([-\d\.]+) z=([-\d\.]+)')
        for line in fp:
            if line.startswith('Total rain is'):
                continue
            mo = re.match(pattern, line)
            if mo is None:
                raise ValueError(repr(line))
            a = int(mo.group(1))
            b = int(mo.group(2))
            for i in range(a, b+1):
                if depths[i] is None:
                    continue
                d1 = depths[a][0] - depths[i][0]
                d2 = depths[a][1] - depths[i][1]
                if d1 > 0.0001:
                    print("%s and %s: File 1 disagrees: %s - %s = %s" %
                          (a, i, depths[a][0], depths[i][0], d1),
                          file=sys.stderr)
                if d2 > 0.0001:
                    print("%s and %s: File 2 disagrees: %s - %s = %s" %
                          (a, i, depths[a][1], depths[i][1], d2),
                          file=sys.stderr)
            r1 = depths[a][2]
            r2 = depths[a][3]
            d = r1 - r2
            area = depths[a][4]
            if tr is None:
                r = '%5s:%5s' % (a, b)
            else:
                r = ','.join(str(tr.get(i, '[%s]' % i)) for i in range(a, b+1))
            print("%g %s %s V=%s A=%s basinVolume=%s ar=%s br=%s az=%s bz=%s" %
                  (d, r, mo.group(3), mo.group(4), area, mo.group(5),
                   r1, r2, depths[a][0], depths[a][1]))


if __name__ == "__main__":
    main()
