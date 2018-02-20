from xml.etree import ElementTree as ET

import ipe.shape
from ipe.shape import Shape, OpaqueShape
from ipe.object import (
    parse_text, parse_image, parse_use, make_group,
    Text, Image, Reference, Group, MatrixTransform, load_matrix,
)


class IpeBitmap:
    def __init__(self, element):
        self.element = element

    def construct(self):
        return self.element


class View:
    def __init__(self, layers, active, marked):
        self.layers = layers
        self.active = active
        self.marked = marked

    @property
    def standard(self):
        return (self.layers == ['alpha'] and
                self.active == 'alpha' and
                not self.marked)

    def __str__(self):
        marked = ' marked' if self.marked else ''
        if len(self.layers) == 1 and self.active == self.layers[0]:
            if self.layers == ['alpha']:
                return '<View%s>' % marked
            else:
                return '<View [%s]%s>' % (self.layers[0], marked)
        else:
            return '<View [%s] active=%s%s>' % (
                ' '.join(self.layers),
                self.active,
                marked)


class IpePage:
    def __init__(self, page, document):
        # ipeiml.cpp ImlParser::parsePage
        self.document = document
        self.page_element = page
        self.objects = []
        self.title = page.get('title', '')
        self.section = page.get('section', '')
        self.subsection = page.get('subsection', '')
        self.marked = page.get('marked') == 'no'
        self.layers = []
        self.views = []
        self.current_layer = None

        for child in page:
            if child.tag == 'notes':
                pass
            elif child.tag == 'layer':
                self.layers.append(child.attrib['name'])
            elif child.tag == 'view':
                self.views.append(View(
                    layers=child.attrib['layers'].split(),
                    active=child.attrib['active'],
                    marked=child.get('marked') == 'yes'))
            else:
                self.current_layer = child.get('layer', self.current_layer)
                if self.current_layer is None:
                    if not self.layers:
                        self.layers.append('alpha')
                    self.current_layer = self.layers[0]
                # ipefactory.cpp createObject
                object = self.parse_object(child, None)
                object.layer = self.current_layer
                self.objects.append(object)
        if not self.views:
            self.views.append(View(
                layers=list(self.layers),
                active=self.layers[0],
                marked=False))

    def __repr__(self):
        return '<IpePage: %s>' % self.summary

    def split_at_view(self, idx):
        first = self.copy()
        second = self.copy()
        first.views = first.views[:idx]
        second.views = second.views[idx:]
        first.prune_objects()
        first.prune()
        second.prune_objects()
        second.prune()
        return first, second

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    @property
    def summary(self):
        t = next(
            (t for view_texts in self.text_strings() for t in view_texts),
            None)
        return '%d objects%s%s%s' % (
            len(self.objects),
            ', %d views' % len(self.views) if len(self.views) > 1 else '',
            ', %d layers' % len(self.layers) if len(self.layers) > 1 else '',
            ', "%s"' % t if t else '')

    def describe(self, indent=''):
        yield indent + self.summary
        view_data = zip(self.views, self.text_strings())
        for i, (view, texts) in enumerate(view_data):
            if len(self.views) != 1 or not view.standard:
                yield indent + 'View %s: %s' % (i, view)
            for t in texts[:1]:
                t_abbr = t if len(t) < 70 else (t[:67] + '...')
                yield indent + '- %s' % t_abbr

    def leaves(self):
        def walk(xs, layer):
            for o in xs:
                if isinstance(o, Group):
                    yield from walk(o.children, layer or o.layer)
                else:
                    yield (layer or o.layer, o)

        return walk(self.objects, None)

    def leaf_objects(self):
        return (o for l, o in self.leaves())

    def itertexts(self):
        return ((l, o) for l, o in self.leaves() if isinstance(o, Text))

    def text_strings(self):
        layer_occurrence = {}
        for i, v in enumerate(self.views):
            for l in v.layers:
                layer_occurrence.setdefault(l, i)
        texts_occurrence = {}
        for g, t in self.itertexts():
            try:
                i = layer_occurrence[g]
            except KeyError:
                continue
            else:
                texts_occurrence.setdefault(i, []).append(t)
        for i, ts in texts_occurrence.items():
            ts.sort(key=lambda t: -t.position[1])
            ts[:] = [' '.join(w for w in t.text.split()
                              if not w.startswith('\\'))
                     for t in ts]
            ts[:] = [t for t in ts if t]
        return [texts_occurrence.get(i, [])
                for i, l in enumerate(self.views)]

    def prune_objects(self):
        "Remove objects in layers that are invisible."
        view_layers = [set(v.layers) for v in self.views]
        visible_layers = set.union(*view_layers)
        self.objects = [
            o for o in self.objects
            if o.layer in visible_layers]

    def prune(self):
        "Remove layers and views that contain no objects."
        nonempty_layers = set(o.layer for o in self.objects)
        if not nonempty_layers:
            print("prune: No objects in page, aborting")
            return
        self.layers = [l for l in self.layers if l in nonempty_layers]
        for v in self.views:
            v.layers = [l for l in self.layers if l in nonempty_layers]
            if v.active not in nonempty_layers:
                if v.layers:
                    v.active = v.layers[0]
                else:
                    v.active = next(iter(nonempty_layers))

    @property
    def lines(self):
        return (o for o in self.objects if o.is_line_segment())

    @property
    def polygons(self):
        return (o for o in self.objects if o.is_polygon())

    def parse_object(self, child, matrix):
        child_matrix = child.attrib.get('matrix')
        if matrix or child_matrix:
            matrix = matrix or MatrixTransform.identity()
            child_matrix = (load_matrix(child_matrix) if child_matrix
                            else MatrixTransform.identity())
            matrix = child_matrix.and_then(matrix)
        if child.tag == 'path':
            s = ipe.shape.load_shape(child.text, child.attrib, matrix)
            if child.attrib.get('matrix'):
                return make_group([s], dict(matrix=child.attrib['matrix']))
            else:
                return s
        elif child.tag == 'text':
            return parse_text(child.text, child.attrib, matrix)
        elif child.tag == 'image':
            if child.get('bitmap'):
                bitmap_id = int(child.attrib['bitmap'])
                bitmap = self.document.bitmaps[bitmap_id]
                return parse_image(bitmap, child.attrib, matrix)
            else:
                return parse_image(child.text, child.attrib, matrix)
        elif child.tag == 'use':
            return parse_use(child.attrib, matrix)
        elif child.tag == 'group':
            return make_group([self.parse_object(c, matrix) for c in child],
                              child.attrib)
        else:
            raise Exception("Unknown tag %s" % (child,))

    def add_shape(self, shape, **kwargs):
        path = ET.SubElement(self.page_element, 'path', **kwargs)
        path.text = '\n' + shape.to_ipe_path()
        path.tail = '\n'

    def add_text(self, text, pos):
        pos_str = '%g %g' % (pos.real, pos.imag)
        t = ET.SubElement(self.page_element, 'text',
            pos=pos_str,
            type='label',
            halign='center',
            valign='center')
        t.text = text
        t.tail = '\n'

    def make_object_element(self, o, **kwargs):
        if isinstance(o, Shape):
            e = ET.Element('path')
            # TODO Compute the right matrix. This is wrong.
            # if o.matrix:
            #     e.set('matrix',
            #           ' '.join('%g' % c for c in o.matrix.coefficients))
            e.text = '\n' + o.to_ipe_path()
            return e
        elif isinstance(o, OpaqueShape):
            e = ET.Element('path', **o.attrib)
            e.text = o.data
            return e
        elif isinstance(o, Text):
            e = ET.Element('text', **o.attrib)
            e.text = o.text
            return e
        elif isinstance(o, Image):
            e = ET.Element('image', **o.attrib)
            bitmaps = kwargs.get('bitmaps', {})
            for i, v in bitmaps.items():
                if v is o.data:
                    e.set('bitmap', str(i))
                    break
            else:
                e.text = o.bitmap_data
            return e
        elif isinstance(o, Reference):
            return ET.Element('use', **o.attrib)
        elif isinstance(o, Group):
            e = ET.Element('group', **o.attrib)
            e.text = '\n'
            for c in o.children:
                ce = self.make_object_element(c, **kwargs)
                assert ce is not None
                ce.tail = '\n'
                e.append(ce)
            return e
        else:
            raise TypeError(type(o))

    def make_page_element(self, bitmaps):
        page = ET.Element('page')
        page.text = '\n'
        for l in self.layers:
            e = ET.SubElement(page, 'layer', name=l)
            e.tail = '\n'
        for v in self.views:
            e = ET.SubElement(page, 'view',
                              active=v.active,
                              layers=' '.join(v.layers))
            e.tail = '\n'
            if v.marked:
                e.set('marked', 'yes')

        self.current_layer = None
        for o in self.objects:
            e = self.make_object_element(o, bitmaps=bitmaps)
            e.tail = '\n'
            if o.layer != self.current_layer:
                self.current_layer = o.layer
                e.set('layer', o.layer)
            page.append(e)
        return page


class IpeDoc:
    def __init__(self, tree):
        self.root = tree.getroot()
        self.bitmaps = {
            int(el.get('id')): IpeBitmap(el)
            for el in self.root.findall('bitmap')
        }
        self.style_names = self.extract_style_names(
            self.root.find('ipestyle'))
        self.pages = [
            IpePage(el, self)
            for el in self.root.findall('page')
        ]

    @property
    def root_attrib(self):
        root_attrib = dict(self.root.attrib)
        root_attrib['creator'] = 'vektorrally.py'
        return root_attrib

    @property
    def info_attrib(self):
        return self.root.find('info').attrib

    @property
    def preamble(self):
        e = self.root.find('preamble')
        if e is None:
            return ''
        else:
            return e.text

    @property
    def ipestyle_element(self):
        return self.root.find('ipestyle')

    def construct(self, bitmaps):
        root = ET.Element(self.root.tag, **self.root_attrib)
        root.text = '\n'
        info = ET.SubElement(root, 'info', **self.info_attrib)
        info.tail = '\n'
        if self.preamble:
            preamble = ET.SubElement(root, 'preamble')
            preamble.text = self.preamble
            preamble.tail = '\n'
        for i in sorted(bitmaps.keys()):
            e = bitmaps[i].construct()
            e.set('id', str(i))
            e.tail = '\n'
            root.append(e)
        root.append(self.ipestyle_element)
        for p in self.pages:
            e = p.make_page_element(bitmaps)
            e.tail = '\n'
            root.append(e)

        for e in root.iter():
            assert e.text is None or isinstance(e.text, str), type(e.text)
            for k, v in e.attrib.items():
                if v is None:
                    continue
                if not isinstance(v, str):
                    raise ValueError((k, v))
        return root

    def save(self, filename):
        with open(filename, 'w') as fp:
            fp.write(
                '<?xml version="1.0"?>\n' +
                '<!DOCTYPE ipe SYSTEM "ipe.dtd">\n')
            fp.write(ET.tostring(self.construct(self.bitmaps),
                                 encoding='unicode'))
            fp.write('\n')

    @staticmethod
    def extract_style_names(ipestyle_element):
        res = {}
        for child in ipestyle_element or ():
            try:
                name = child.attrib['name']
            except KeyError:
                continue
            res.setdefault(child.tag, set()).add(name)
        return res

    def __repr__(self):
        return "<IpeDoc: %d pages %s>" % (len(self.pages), self.pages)

    def describe(self):
        for i, page in enumerate(self.pages):
            yield 'Page %d:' % i
            yield from page.describe('  ')

    def split_page_at_view(self, page_idx, view_idx):
        a, b = self.pages[page_idx].split_at_view(view_idx)
        self.pages[page_idx:page_idx+1] = [a, b]


def parse(filename):
    return IpeDoc(ET.parse(filename))
