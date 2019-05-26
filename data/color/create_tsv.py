import sqlite3 as sql
from collections import Counter, defaultdict, namedtuple

class Color:

    def __init__(self, r, g, b, name=None):
        self.r = r
        self.g = g
        self.b = b
        self.name = name

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.to_hex()

    def __repr__(self):
        if self.name is not None:
            return 'Color({}, {}, {}, name={})'.format(self.r, self.g, self.b, repr(self.name))
        else:
            return 'Color({}, {}, {})'.format(self.r, self.g, self.b)

    def __eq__(self, other):
        return str(self) == str(other)

    def __sub__(self, other):
        return abs(self.r - other.r) + abs(self.g - other.g) + abs(self.b - other.b)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return [self.r, self.g, self.b][index]

    def to_hex(self):
        return '#{:02x}{:02x}{:02x}'.format(self.r, self.g, self.b).upper()

    @staticmethod
    def from_hex(hexcode, name=None):
        if len(hexcode) == 7 and hexcode[0] == '#':
            hexcode = hexcode[1:]
        return Color(*(int(hexcode[i:i + 2], 16) for i in range(0, 5, 2)), name=name)

QUERY = 'SELECT colorname, r, g, b FROM answers;'

Answer = namedtuple('Answer', ('name', 'r', 'g', 'b'))

conn = sql.connect('color-survey.sqlite')
conn.row_factory = (lambda cursor, row: Answer(*row))
cursor = conn.cursor()

colors = defaultdict(Counter)
for answer in cursor.execute(QUERY):
    colors[answer.name.replace('-', ' ')].update([Color(answer.r, answer.g, answer.b)])
centroids = {}
for name, rgbs in colors.items():
    count = sum(rgbs.values())
    r = round(sum(value.r * count for value, count in rgbs.items()) / count)
    g = round(sum(value.g * count for value, count in rgbs.items()) / count)
    b = round(sum(value.b * count for value, count in rgbs.items()) / count)
    centroids[name] = Color(r, g, b)
with open('color-centroids.tsv', 'w') as fd:
    for name, rgbs in sorted(colors.items(), key=(lambda kv: len(kv[1])), reverse=True):
        count = sum(rgbs.values())
        if count >= 1000:
            centroid = centroids[name]
            fd.write('{}\t{}\t{}\n'.format(count, name, centroid))
