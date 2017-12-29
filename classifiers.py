from ast import literal_eval
from collections import defaultdict, Counter
from itertools import groupby
from os.path import exists as file_exists, join as join_path
from random import shuffle
from statistics import mean

FILE_DIR = dirname(realpath(__file__))


class DomainUtils:

    @classmethod
    def class_to_label(cls, y):
        """Convert an integer class to a human-readable label

        Arguments:
            y (int): the integer class

        Returns:
            str: the human-readable label
        """
        raise NotImplementedError()

    @classmethod
    def label_to_class(cls, label):
        """Convert an integer class to a human-readable label

        Arguments:
            label (str): the human-readable label

        Returns:
            int: the integer class
        """
        raise NotImplementedError()

    @classmethod
    def class_distance(cls, y1, y2):
        """Calculate the distance between two classes

        Arguments:
            y1 (int): the first class
            y2 (int): the second class

        Returns:
            float: the distance between the classes
        """
        raise NotImplementedError()

    @classmethod
    def label_distance(cls, label1, label2):
        """Calculate the distance between two labels

        Arguments:
            label1 (str): the first label
            label2 (str): the second label

        Returns:
            float: the distance between the classes
        """
        raise NotImplementedError()


class Classifier:

    def get_persistent_id(self):
        """Get a persistent (string) id

        Returns:
            str: an id for this classifier
        """
        raise NotImplementedError()

    def get_ys(self):
        """Get the labels

        Returns:
            [int]: the integer labels
        """
        raise NotImplementedError()

    def classify(self, xs):
        """Classify the data

        Arguments:
            xs: the data to be classified
                this can be whatever Dataset.get_x() returns

        Returns:
            [int]: the classifications
        """
        raise NotImplementedError()

    def to_file(self, filepath):
        """Save the classifier to file

        Arguments:
            filepath (str): The file to save to
        """
        raise NotImplementedError()

    @classmethod
    def from_file(cls, filepath):
        """Create a classifier from file

        Arguments:
            filepath (str): The file to load from

        Returns:
            Classifier: 
        """
        raise NotImplementedError()


class Dataset:

    def get_persistent_id(self):
        raise NotImplementedError()

    def get_x(self):
        raise NotImplementedError()

    def get_y(self):
        raise NotImplementedError()


class RegretTrial:

    def __init__(self, classifier, domain_utils, dataset, path_prefix='cache'):
        self.classifier = classifier
        self.domain_utils = domain_utils
        self.dataset = dataset
        self.regrets = self.load_regrets()
        self.path_prefix = path_prefix

    def get_persistent_id(self):
        classifier_id = self.classifier.get_persistent_id()
        dataset_id = self.dataset.get_persistent_id()
        return join_path(FILE_DIR, self.path_prefix, '{}_{}'.format(classifier_id, dataset_id))

    def all_ys(self):
        return self.old_ys() + self.new_ys()

    def old_ys(self):
        return self.classifier.get_ys()

    def new_ys(self):
        return list(set(self.dataset.get_y()))

    def get_regrets_file(self):
        return self.get_persistent_id() + '.regrets'

    def load_regrets(self):
        if not file_exists(self.get_regrets_file()):
            with open(self.get_regrets_file(), 'w') as fd:
                fd.write('{\n')
                for new_y in sorted(self.new_ys()):
                    fd.write(repr(new_y) + ':{\n')
                    fd.write('    {}:{},\n'.format(repr('mean_regret'), self._calculate_label_mean_regret(new_y)))
                    fd.write('    {}:{},\n'.format(repr('max_regret'), self._calculate_label_max_regret(new_y)))
                    fd.write('},\n')
                fd.write('}\n')
        with open(self.get_regrets_file()) as fd:
            return literal_eval(fd.read())

    def _calculate_label_mean_regret(self, new_y):
        misclassifications = self.label_misclassification_order(new_y)
        distances = self.label_distance_order(new_y)
        return ranking_regret(misclassifications, distances)

    def _calculate_label_max_regret(self, new_y):
        misclassifications = self.label_misclassification_order(new_y)
        worst_case_heuristic = [[y, -score] for y, score in misclassifications]
        return ranking_regret(misclassifications, worst_case_heuristic)

    def label_misclassification_order(self, new_y):
        """The order of old labels by decreasing false positives of the new label.

        Note the score is negated so that a standard sort will work.

        Arguments:
            new_y (str):
                The label to compute the misclassification order for.

        Returns:
            [[str, int]]: the old labels and their (negated) false positives
        """

        misclassifications = []
        for old_y, counter in self.load_summary().items():
            if new_y in counter:
                misclassifications.append([old_y, -counter[new_y]])
            else:
                misclassifications.append([old_y, 0])
        # lambda sorts by num misclassified, then by label name
        return sorted(misclassifications, key=(lambda kv: list(reversed(kv))))

    def label_distance_order(self, new_y):
        """The order of old labels by increasing heuristic distance.

        Arguments:
            new_y (str):
                The label to compute the distance order for.

        Returns:
            [[str, int]]: an generator of lists of orderings
        """

        # FIXME this is no longer correct because we separate y and label
        '''
        if 'FIXME' in new_label:
            return None
        '''
        distances = [[old_y, self.domain_utils.class_distance(old_y, new_y)] for old_y in self.old_ys()]
        # lambda sorts by distance, then by label name
        return sorted(distances, key=(lambda kv: list(reversed(kv))))

    def label_accuracy(self, old_label):
        summary = self.load_summary()
        return summary[old_label].loc[old_label] / sum(summary[old_label])

    def label_random_regret(self, new_label):
        misclassifications = self.label_misclassification_order(new_label)
        shuffled_scores = list(range(len(misclassifications)))
        regrets = []
        for _ in range(100):
            shuffle(shuffled_scores)
            random_order = list(zip([kv[0] for kv in misclassifications], shuffled_scores))
            regret = ranking_regret(misclassifications, random_order)
            regrets.append(regret)
        return mean(regrets)

    def label_mean_regret(self, new_label):
        return self.regrets[new_label]['mean_regret']

    def label_max_regret(self, new_label):
        return self.regrets[new_label]['max_regret']

    def get_summary_file(self):
        return self.get_persistent_id() + '.summary'

    def load_summary(self):
        """Load, or calculate and cache, a summary of this trial

        Returns:
            {y_hat: {y: int}}: a summary of the classification distribution
        """
        if not file_exists(self.get_summary_file()):
            # create "table" of predictions
            counter_stats = defaultdict(Counter)
            for y_hat, y in zip(self.load_predictions(), self.dataset.get_y()):
                counter_stats[y_hat].update([y])
            # convert into normal dictionary
            stats = {}
            for y_hat, counter in counter_stats.items():
                stats[y_hat] = dict(counter)
            with open(self.get_summary_file(), 'w') as fd:
                fd.write('{\n')
                for y_hat, counter in sorted(stats.items()):
                    fd.write('    {}: {},\n'.format(repr(y_hat), repr(counter)))
                fd.write('}\n')
        with open(self.get_summary_file()) as fd:
            return literal_eval(fd.read())

    def get_predictions_file(self):
        return self.get_persistent_id() + '.predictions'

    def load_predictions(self):
        """Load, or calculate and cache, the classifications of this dataset

        Returns:
            [y_hat]: a summary of the classification distribution
        """
        if not file_exists(self.get_predictions_file()):
            predictions = self.classifier.classify(self.dataset.get_x())
            with open(self.get_predictions_file(), 'w') as fd:
                fd.write('[\n')
                for y in predictions:
                    fd.write('    {},\n'.format(repr(y)))
                fd.write(']\n')
        with open(self.get_predictions_file()) as fd:
            return literal_eval(fd.read())


def bucket_alist(alist):
    """Group keys in an association list by their values

    Arguments:
        alist [[str, int]]: A list of [str, int] association pairs

    Returns:
        [int, [str]]: A list of grouped keys by their values
    """
    values = dict(alist)
    keys = sorted(values.keys(), key=(lambda k: values[k]))
    bucketed = list((v, list(ks)) for v, ks in groupby(keys, key=(lambda k: values[k])))
    return bucketed


def intra_bucket_regret(bucket, groundtruth):
    """Calculate regret of one bucket.

    Arguments:
        bucket [str]: A list of labels
        groundtruth {str:int}: Label to misclassifications dictionary

    Returns:
        double: regret of the labels in the bucket
    """
    groundtruth_ranking = sorted(
        ([label, groundtruth[label]] for label in bucket),
        key=(lambda kv: kv[1]),
    )
    regret = 0
    for i, [_, score_1] in enumerate(groundtruth_ranking):
        for _, score_2 in groundtruth_ranking[i + 1:]:
            if score_1 < score_2:
                regret += score_2 - score_1
    return regret / 2


def inter_bucket_regret(bucket_1, bucket_2, groundtruth):
    """Calculate regret between two buckets.

    Arguments:
        bucket_1 [str]: Labels in the first bucket
        bucket_2 [str]: Labels in the second bucket
        groundtruth {str:int}: Label to misclassifications dictionary

    Returns:
        double: regret of the labels due to between-bucket scores
    """
    regret = 0
    for label_1 in bucket_1:
        for label_2 in bucket_2:
            if groundtruth[label_2] < groundtruth[label_1]:
                regret += groundtruth[label_1] - groundtruth[label_2]
    return regret


def ranking_regret(groundtruth_ranking, heuristic_ranking):
    """Calculate regret between bucket orders.

    Arguments:
        groundtruth_ranking ([[str, int]]):
            The correct ranks of the labels
        heuristic_ranking ([[str, int]]):
            A heuristic ranks of the labels

    Returns:
        double: the mean regret over all total orders of the heuristic ranking
    """
    groundtruth = dict(groundtruth_ranking)
    heuristic_bucketed = bucket_alist(heuristic_ranking)
    regret = 0
    for i, bucket_1 in enumerate(heuristic_bucketed):
        regret += intra_bucket_regret(bucket_1[1], groundtruth)
        for bucket_2 in heuristic_bucketed[i + 1:]:
            regret += inter_bucket_regret(bucket_1[1], bucket_2[1], groundtruth)
    return regret
