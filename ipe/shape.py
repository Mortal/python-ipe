import re
from .object import IpeObject, load_matrix, MatrixTransform


class Curve:
    def __init__(self, edges=None, closed=False, matrix=None):
        self.edges = [] if edges is None else list(edges)
        self.closed = closed
        self.matrix = matrix

    def append_segment(self, u, v):
        self.edges.append((u, v))

    def __repr__(self):
        return "Curve(edges=%r, closed=%r, matrix=%r)" % (
            self.edges, self.closed, self.matrix)

    def __bool__(self):
        return bool(self.edges)

    @property
    def mass_middle(self):
        masses = []
        mids = []
        for u, v in self.get_edges():
            uv = v - u
            mids.append(u + uv / 2)
            masses.append(abs(uv))
        mass = sum(masses)
        middle = sum(m * (l / mass) for m, l in zip(mids, masses))
        return mass, middle

    @property
    def position(self):
        mass, middle = self.mass_middle
        return middle.real, middle.imag

    def move(self, d):
        self.edges = [(u+d, v+d) for u, v in self.edges]

    @classmethod
    def make_polyline(cls, points):
        if not all(isinstance(p, complex) for p in points):
            raise TypeError(str(points))
        edges = [
            (u, v)
            for u, v in zip(points[:-1], points[1:])
        ]
        return cls(edges)

    def get_edges(self):
        edges = list(self.edges)
        if self.closed:
            edges.append((edges[-1][1], edges[0][0]))
        return [self.transform_edge(e) for e in edges]

    def transform_edge(self, e):
        u, v = e
        return self.transform_point(u), self.transform_point(v)

    def transform_point(self, v):
        if self.matrix is None:
            return v
        x, y = v.real, v.imag
        xp, yp = self.matrix.transform(x, y)
        return complex(xp, yp)

    def is_line_segment(self):
        return not self.closed and len(self.edges) == 1

    def is_polygon(self):
        return self.closed

    def is_triangle(self):
        return self.is_polygon() and len(self.edges) == 2

    def endpoints(self):
        return self.transform_edge(self.edges[0])

    def to_ipe_path(self):
        directives = []
        pos = None
        for u, v in self.edges:
            if pos is None:
                pos = u
                directives.append('%g %g m\n' % (u.real, u.imag))
            if pos != u:
                raise Exception("to_ipe_path: Edges are not connected")
            pos = v
            directives.append('%g %g l\n' % (v.real, v.imag))
        if self.closed:
            directives.append('h\n')
        return ''.join(directives)


class Shape(IpeObject):
    def __init__(self, curves, matrix):
        super().__init__(matrix=matrix)
        self.curves = curves

    def __repr__(self):
        return "Shape(%r)" % (self.curves,)

    @property
    def mass_middle(self):
        masses, mids = zip(*[c.mass_middle for c in self.curves])
        mass = sum(masses)
        middle = sum(m * (l / mass) for m, l in zip(mids, masses))
        return mass, middle

    @property
    def position(self):
        mass, middle = self.mass_middle
        return self.transform((middle.real, middle.imag))

    def move(self, dx, dy):
        ox, oy = self.reverse_transform((0, 0))
        ax, ay = self.reverse_transform((dx, dy))
        print('From %s to %s in world is from %s to %s in object' %
              ((0, 0), (dx, dy), (ox, oy), (ax, ay)))
        d = complex(ax - ox, ay - oy)
        for c in self.curves:
            c.move(d)

    @classmethod
    def make_polyline(cls, points):
        return cls([Curve.make_polyline(points)])

    @property
    def is_line_segment(self):
        return len(self.curves) == 1 and self.curves[0].is_line_segment()

    @property
    def is_polygon(self):
        return len(self.curves) == 1 and self.curves[0].is_polygon()

    @property
    def is_triangle(self):
        return len(self.curves) == 1 and self.curves[0].is_triangle()

    def get_edges(self):
        return [e for c in self.curves for e in c.get_edges()]

    def endpoints(self):
        return self.curves[0].endpoints()

    def to_ipe_path(self):
        return ''.join(c.to_ipe_path() for c in self.curves)


class OpaqueShape(IpeObject):
    def __init__(self, data, attrib, matrix):
        self.data = data
        self.attrib = attrib
        super().__init__(matrix=matrix)

    @property
    def position(self):
        mo = re.search(r'(\d+\.?\d*) (\d+\.?\d*)', self.data)
        if not mo:
            raise ValueError(self.data)
        x, y = float(mo.group(1)), float(mo.group(2))
        return self.transform((x, y))

    def __repr__(self):
        return '<OpaqueShape %r>' % re.sub(r'[^a-z]', '', self.data)


def load_shape(data, attrib=None, matrix=None):
    """Load Ipe shape data from a string.

    Mirrors ipe/src/ipelib/ipeshape.cpp Shape::load.
    """

    if attrib is None:
        attrib = dict()
    if matrix is None and attrib.get('matrix'):
        matrix = load_matrix(attrib.get('matrix'))

    curves = []
    args = []  # list of coordinates
    subpath = None
    current_position = 0j

    def pop_point():
        y, x = args.pop(), args.pop()
        return complex(x, y)

    def pop_matrix():
        m = MatrixTransform(args[-6:])
        del args[-6:]
        return m

    for tok in data.split():
        if tok == "h":
            # Closing path
            if not subpath:
                raise ValueError()
            subpath.closed = True
            subpath = None
        elif tok == "m":
            if len(args) != 2:
                raise ValueError()
            subpath = Curve(matrix=matrix)
            curves.append(subpath)
            current_position = pop_point()
        elif tok == "l":
            if len(args) != 2:
                raise ValueError()
            v = pop_point()
            subpath.append_segment(current_position, v)
            current_position = v
        elif tok in 'e q c a s u'.split():
            return OpaqueShape(data, attrib, matrix)
        else:
            # Must be a number
            args.append(float(tok))

    if len(curves) == 0:
        raise ValueError()
    if not all(curves):
        raise ValueError()
    return Shape(curves, matrix)


def apply_matrix_to_shape(data, matrix):
    result = ['\n']
    x = None
    for tok in data.split():
        if tok in 'h m l q c a s e u'.split():
            result.append('%s\n' % tok)
        elif x is None:
            x = float(tok)
        else:
            y = float(tok)
            result.append('%s %s ' % matrix.transform(x, y))
            x = None
    return ''.join(result)
