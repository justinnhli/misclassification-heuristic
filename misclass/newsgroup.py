from ast import literal_eval
from itertools import combinations
from random import Random

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from misclass import DomainUtils, Classifier, Dataset, RegretTrial

LABELS = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc',
]

class NewsgroupUtils(DomainUtils):

    def __init__(self, int_labels):
        self.int_labels = int_labels

    def class_to_label(self, y):
        return LABELS[y]

    def label_to_class(self, label):
        return LABELS.index(label)

    def class_distance(self, y1, y2):
        return self.label_distance(self.class_to_label(y1), self.class_to_label(y2))

    def label_distance(self, label1, label2):
        path1 = label1.split('.')
        path2 = label2.split('.')
        common = sum(1 for part1, part2 in zip(path1, path2) if part1 == part2)
        return len(path1) + len(path2) - (2 * common)


class NewsgroupDataset(Dataset):

    def __init__(self, int_labels):
        self.int_labels = sorted(int_labels)
        self.data = fetch_20newsgroups(
            subset='train',
            remove=('headers', 'footers', 'quotes'),
            categories=[LABELS[i] for i in self.int_labels],
        )

    def get_persistent_id(self):
        return str(ints_to_binary(self.int_labels))

    def get_x(self):
        return self.data.data

    def get_y(self):
        return [
            LABELS.index(self.data.target_names[i])
            for i in self.data.target
        ]


class NewsgroupClassifier(Classifier):

    def __init__(self, int_labels):
        self.int_labels = int_labels
        self.pipeline = Pipeline([
            ('vect', CountVectorizer(max_df=0.75, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(tol=1e-3, alpha=1e-05, max_iter=20, penalty='elasticnet')),
        ])
        dataset = NewsgroupDataset(self.int_labels)
        self.pipeline.fit(dataset.get_x(), dataset.get_y())
        # FIXME should save/pickle the pipeline

    def get_persistent_id(self):
        return str(ints_to_binary(self.int_labels))

    def get_ys(self):
        return self.int_labels

    def classify(self, xs):
        return self.pipeline.predict(xs)

    def to_file(self, filepath):
        raise NotImplementedError()

    @classmethod
    def from_file(cls, filepath):
        raise NotImplementedError()


def ints_to_binary(int_labels):
    """
    Arguments:
        int_labels ([int]): The index of the 1 bits

    Returns:
        int: The binary as an integer
    """
    return sum(2**i for i in int_labels)


def binary_to_ints(binary):
    """
    Arguments:
        binary (int): The binary as an integer

    Returns:
        [int]: The index of the 1 bits
    """
    binary_str = '{:b}'.format(binary)
    return sorted(i for i, b in enumerate(reversed(binary_str)) if b == '1')

def main():
    rng = Random(8675309)
    for num in range(2, 20):
        for orig_labels in combinations(list(range(20)), num):
            classifier = NewsgroupClassifier(orig_labels)
            int_labels = list(range(20))
            utils = NewsgroupUtils(int_labels)
            dataset = NewsgroupDataset(int_labels)
            trial = RegretTrial(classifier, utils, dataset, path_prefix='newsgroup_cache')

if __name__ == '__main__':
    main()
