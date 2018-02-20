import os
import pprint
import argparse
from IPython.terminal.embed import InteractiveShellEmbed
from .file import parse


def style_sets(files, tag):
    return [f.style_names.get(tag, set()) for f in files]


def check_styles(files):
    tags = sorted(set(t for f in files for t in f.style_names.keys()))
    for t in tags:
        sets = style_sets(files, t)
        union = set.union(*sets)
        intersection = set.intersection(*sets)
        difference = sorted(union - intersection)
        if difference:
            print("Not all files have defined the following names for %s: %s" %
                  (t, ' '.join(difference)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    args = parser.parse_args()

    basenames = [os.path.splitext(os.path.basename(f))[0]
                 for f in args.filenames]
    files = [parse(f) for f in args.filenames]
    vars = dict(parse=parse)
    pages_str = []
    for filename, basename, f in zip(args.filenames, basenames, files):
        varname = basename.replace('-', '_')
        vars[varname] = f
        for i, p in enumerate(f.pages):
            pages_str.append('%s.pages[%r] == %r' % (varname, i, p))

    check_styles(files)

    ipshell = InteractiveShellEmbed(user_ns=vars)
    ipshell('\n'.join(pages_str))


if __name__ == "__main__":
    main()
