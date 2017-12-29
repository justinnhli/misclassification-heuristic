#!/usr/bin/env python3

import re
from os.path import basename, dirname, expanduser, join as join_path, realpath, splitext
from random import seed as set_seed, randrange

from classifiers import DomainUtils, Classifier, Dataset, RegretTrial

DIRECTORY = dirname(realpath(__file__))
DIRECTORY = realpath(expanduser('~/git/aaai2018/color-code'))
COLOR_NAMES_FILE = join_path(DIRECTORY, 'color-centroids.tsv')


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

    def to_hex(self):
        return '#{:02x}{:02x}{:02x}'.format(self.r, self.g, self.b).upper()

    @staticmethod
    def from_hex(hexcode, name=None):
        if len(hexcode) == 7 and hexcode[0] == '#':
            hexcode = hexcode[1:]
        return Color(*(int(hexcode[i:i + 2], 16) for i in range(0, 5, 2)), name=name)


def read_colors():
    colors = []
    with open(COLOR_NAMES_FILE) as fd:
        for line in fd:
            line = line.strip()
            if line[0] == '#':
                continue
            _, name, hexcode = line.split('\t')
            assert name, 'Colors must have a name, but only got hex: {}'.format(hexcode)
            colors.append(Color.from_hex(hexcode, name))
    return colors


CENTROIDS = read_colors()


class ColorUtils(DomainUtils):

    def __init__(self, centroids):
        self.centroids = centroids

    def class_to_color(self, y):
        return self.centroids[y]

    def label_to_color(self, label):
        candidates = [color for color in self.centroids if color.name == label]
        assert len(candidates) == 1
        return candidates[0]

    def class_to_label(self, y):
        return self.class_to_color(y).name

    def label_to_class(self, label):
        return self.label_to_color(label).name

    def class_distance(self, y1, y2):
        return abs(self.class_to_color(y1) - self.class_to_color(y2))

    def label_distance(self, label1, label2):
        return abs(self.class_to_label(label1) - self.class_to_label(label2))


class NearestCentroid(Classifier):

    def __init__(self, centroids, label_map):
        """Constructor for a nearest centroid classifier

        Arguments:
            centroids ([Color]): the centroid colors
        """
        self.centroids = centroids
        self.label_map = label_map

    def get_persistent_id(self):
        """Get a persistent (string) id

        Returns:
            str: an id for this classifier
        """
        return 'colors{}'.format(len(self.centroids))

    def get_ys(self):
        """Get the labels

        Returns:
            [int]: the integer labels
        """
        return list(range(len(self.centroids)))

    def classify(self, xs):
        """Classify the data

        Arguments:
            xs: the data to be classified
                this can be whatever Dataset.get_x() returns

        Returns:
            [int]: the classifications
        """
        ys = []
        for x in xs:
            closest_index = min(
                range(len(self.centroids)),
                key=(lambda k, x=x: abs(x - self.centroids[k])),
            )
            ys.append(closest_index)
        return ys

    def to_file(self, filepath):
        """Save the classifier to file

        Arguments:
            filepath (str): The file to save to
        """
        with open(filepath, 'w') as fd:
            for centroid in self.centroids:
                line = '{}\t{}\n'.format(centroid.to_hex(), centroid.name)
                fd.write(line)

    @classmethod
    def from_file(cls, filepath):
        """Create a classifier from file

        Arguments:
            filepath (str): The file to load from
        """
        centroids = []
        with open(filepath) as fd:
            for line in fd.readlines():
                line = line.strip()
                hex_code, name = line.split('\t')
                centroids.append(Color.from_hex(hex_code, name))
        return cls(centroids, ColorUtils(centroids))


class ColorDataset(Dataset):

    def __init__(self, size, random_seed, num_colors, label_map):
        self.size = size
        self.seed = random_seed
        self.num_colors = num_colors
        self.label_map = label_map
        self.classifier = NearestCentroid(CENTROIDS[:self.num_colors], self.label_map)
        self.x = None
        self.y = None
        self.labels = None

    def get_persistent_id(self):
        return 's{}n{}k{}'.format(self.seed, self.size, self.num_colors)

    def get_x(self):
        if self.x is None:
            self.x = self._random_colors()
        return self.x

    def get_y(self):
        if self.y is None:
            self.y = self.classifier.classify(self.get_x())
        return self.y

    def get_labels(self):
        if self.labels is None:
            self.labels = self.label_map.get_lable(self.get_y())
        return self.labels

    def _random_colors(self):
        set_seed(self.seed)
        result = []
        for _ in range(self.size):
            result.append(Color(randrange(256), randrange(256), randrange(256)))
        return result


def from_file(filepath):
    filename = splitext(basename(filepath))[0]
    classifier_id, dataset_id = filename.split('_', maxsplit=1)
    # parse parameters
    classifier_match = re.match('colors(?P<k>[0-9]+)', classifier_id)
    num_old_colors = int(classifier_match.group('k'))
    dataset_match = re.match('s(?P<s>[0-9]+)n(?P<n>[0-9]+)k(?P<k>[0-9]+)', dataset_id)
    random_seed = int(dataset_match.group('s'))
    num_samples = int(dataset_match.group('n'))
    num_new_colors = int(dataset_match.group('k'))
    # create classifier, dataset, and regret trial
    classifier = NearestCentroid(
        CENTROIDS[:num_old_colors],
        ColorUtils(CENTROIDS[:num_old_colors]),
    )
    dataset = ColorDataset(
        num_samples,
        random_seed,
        num_new_colors,
        ColorUtils(CENTROIDS[:num_new_colors]),
    )
    return RegretTrial(
        classifier,
        ColorUtils(CENTROIDS[:num_new_colors]),
        dataset,
    )


def main():
    NUM_OLD_COLORS = 10
    NUM_NEW_COLORS = 20
    NUM_TEST_COLORS = 1000
    # create classifier (old labels)
    classifier = NearestCentroid(
        CENTROIDS[:NUM_OLD_COLORS],
        ColorUtils(CENTROIDS[:NUM_OLD_COLORS]),
    )
    # create regret testing data (new labels)
    random_seed = 8675309 # FIXME
    dataset = ColorDataset(
        NUM_TEST_COLORS,
        random_seed,
        NUM_NEW_COLORS,
        ColorUtils(CENTROIDS[:NUM_NEW_COLORS]),
    )
    # create regret trial
    RegretTrial(
        classifier,
        ColorUtils(CENTROIDS[:NUM_NEW_COLORS]),
        dataset,
        path_prefix='colors',
    )


if __name__ == '__main__':
    main()
