import os
import random
import shutil
import argparse
import contextlib
import collections

import shapefile


Field = collections.namedtuple('Field', 'name dtype size decimals')


@contextlib.contextmanager
def iterdbf(filename):
    assert filename.endswith('.dbf')
    with open(filename, 'rb') as fp:
        reader = shapefile.Reader(dbf=fp)
        fields = reader.fields
        if fields[0] == ('DeletionFlag', 'C', 1, 0):
            fields = fields[1:]
        assert all(f[0] != 'DeletionFlag' for f in fields)
        fields = [Field(*f) for f in fields]
        yield fields, reader.iterRecords()


def write_dbf(filename, fields, records):
    assert filename.endswith('.dbf')
    writer = shapefile.Writer()
    for f in fields:
        writer.field(*f)
    for r in records:
        assert len(r) == len(fields)
        writer.record(*r)
    with open(filename, 'w+b') as fp:
        writer.saveDbf(fp)


def clip(v, lo, hi):
    return type(v)(min(hi, max(lo, v)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--clobber', action='store_true')
    parser.add_argument('-n', '--no-clobber', action='store_true')
    parser.add_argument('-i', '--init')
    parser.add_argument('-e', '--expression', required=True)
    parser.add_argument('filename')
    args = parser.parse_args()
    if args.clobber and args.no_clobber:
        parser.error("Must not specify both -n and -f")
    backup_name = args.filename + '.bak'
    if os.path.exists(backup_name) and not (args.clobber or args.no_clobber):
        parser.error("Specify -f to overwrite or -n to not overwrite backup file %r" % backup_name)

    try:
        compile_expr = compile(args.expression, '<-e>', 'eval')
    except Exception as exn:
        parser.error(str(exn))

    with iterdbf(args.filename) as (fields, iterrecords):
        records = list(iterrecords)
    field_dict = {f.name: f for f in fields}

    extra = []
    environ = dict(random=random, clip=clip)
    if args.init:
        exec(args.init, environ)
    for r in records:
        expr_environ = dict(environ)
        expr_environ.update({f.name: v for f, v in zip(fields, r)})
        result = eval(compile_expr, expr_environ) or {}
        if not isinstance(result, dict):
            parser.error("Expression must return a dict, not %s" %
                         (type(result).__name__,))
        for k, v in result.items():
            if len(k) > 11:
                parser.error('Field name must be at most 11 characters: %r' % (k,))
            if isinstance(v, float):
                field_definition = Field(k, 'F', 25, 10)
            elif isinstance(v, int):
                field_definition = Field(k, 'N', 9, 0)
            elif v is None:
                continue
            else:
                raise NotImplementedError(type(v).__name__)
            current_definition = field_dict.setdefault(k, field_definition)
            if current_definition != field_definition:
                parser.error("Incompatible definitions: %r != %r" %
                             (current_definition, field_definition))
        extra.append(result)

    preexisting = set(f.name for f in fields)
    extra_count = len(field_dict) - len(fields)
    output_fields = fields + [field_dict[k] for k in field_dict.keys() - preexisting]
    field_index = {f.name: i for i, f in enumerate(output_fields)}
    output_records = []
    for r, e in zip(records, extra):
        output_record = list(r) + [None]*extra_count
        for k, v in e.items():
            output_record[field_index[k]] = v
        output_records.append(output_record)
    t = '.~%s' % args.filename
    write_dbf(t, output_fields, output_records)
    if not os.path.exists(backup_name) or args.clobber:
        os.rename(args.filename, backup_name)
    os.rename(t, args.filename)


if __name__ == '__main__':
    main()
