import re
from ast import literal_eval
from collections import defaultdict, Counter, namedtuple
from datetime import datetime
from itertools import groupby
from os import remove as rm, makedirs, listdir
from os.path import join as join_path, exists as file_exists, realpath, expanduser, basename, splitext, dirname
from random import shuffle
from statistics import mean

import numpy as np
import pandas as pd

from images import image_from_file, NeuralNetwork
from colors import color_from_file


FILE_DIR = dirname(realpath(__file__))


class DomainUtils:

    @classmethod
    def class_to_label(cls, y):
        """Convert an integer class to a human-readable label.

        Arguments:
            y (int): the integer class.

        Returns:
            str: the human-readable label.
        """
        raise NotImplementedError()

    @classmethod
    def label_to_class(cls, label):
        """Convert an integer class to a human-readable label.

        Arguments:
            label (str): the human-readable label.

        Returns:
            int: the integer class.
        """
        raise NotImplementedError()

    @classmethod
    def class_distance(cls, y1, y2):
        """Calculate the distance between two classes.

        Arguments:
            y1 (int): the first class.
            y2 (int): the second class.

        Returns:
            float: the distance between the classes.
        """
        raise NotImplementedError()

    @classmethod
    def label_distance(cls, label1, label2):
        """Calculate the distance between two labels.

        Arguments:
            label1 (str): the first label.
            label2 (str): the second label.

        Returns:
            float: the distance between the classes.
        """
        raise NotImplementedError()


class Classifier:

    def get_persistent_id(self):
        """Get a persistent (string) id.

        Returns:
            str: an id for this classifier.
        """
        raise NotImplementedError()

    def get_ys(self):
        """Get the labels.

        Returns:
            [int]: the integer labels.
        """
        raise NotImplementedError()

    def classify(self, xs):
        """Classify the data.

        Arguments:
            xs (list[object]): The data to be classified.
                This can be whatever Dataset.get_x() returns.

        Returns:
            [int]: the classifications.
        """
        raise NotImplementedError()

    def to_file(self, filepath):
        """Save the classifier to file.

        Arguments:
            filepath (str): The file to save to.
        """
        raise NotImplementedError()

    @classmethod
    def from_file(cls, filepath):
        """Create a classifier from file.

        Arguments:
            filepath (str): The file to load from.

        Returns:
            Classifier: The classifier.
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
        self.path_prefix = path_prefix
        makedirs(self.get_data_path(), exist_ok=True)
        self.regrets = self.load_regrets()

    def get_persistent_id(self):
        classifier_id = self.classifier.get_persistent_id()
        dataset_id = self.dataset.get_persistent_id()
        return classifier_id + '_' + dataset_id

    def get_data_path(self):
        return join_path(FILE_DIR, self.path_prefix)

    def all_ys(self):
        return self.old_ys() + self.new_ys()

    def old_ys(self):
        return self.classifier.get_ys()

    def new_ys(self):
        return list(set(self.dataset.get_y()))

    def get_regrets_file(self):
        return join_path(self.get_data_path(), self.get_persistent_id() + '.regrets')

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
        try:
            with open(self.get_regrets_file()) as fd:
                cached = literal_eval(fd.read())
        except SyntaxError:
            rm(self.get_regrets_file())
            cached = self.load_regrets()
        return cached

    def _calculate_label_mean_regret(self, new_y):
        misclassifications = self.label_misclassification_order(new_y)
        distances = self.label_distance_order(new_y)
        return ranking_regret(misclassifications, distances)

    def _calculate_label_max_regret(self, new_y):
        misclassifications = self.label_misclassification_order(new_y)
        worst_case_heuristic = [[y, -score] for y, score in misclassifications]
        return ranking_regret(misclassifications, worst_case_heuristic)

    def label_misclassification_order(self, new_y):
        """Get the order of old labels by decreasing false positives of the new label.

        Note the score is negated so that a standard sort will work.

        Arguments:
            new_y (str):
                The label to compute the misclassification order for.

        Returns:
            List[(str, int)]: The old labels and their (negated) false positives.
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
        """Get the order of old labels by increasing heuristic distance.

        Arguments:
            new_y (str):
                The label to compute the distance order for.

        Returns:
            List[(str, int)]: an generator of lists of orderings
        """
        distances = [[old_y, self.domain_utils.class_distance(old_y, new_y)] for old_y in self.old_ys()]
        # lambda sorts by distance, then by label name
        return sorted(distances, key=(lambda kv: list(reversed(kv))))

    def label_true_positive_init(self, old_label):
        if hasattr(self.classifier, 'label_true_positive_init'):
            return self.classifier.label_true_positive_init(old_label)
        else:
            return np.nan

    def true_positive_init(self):
        if hasattr(self.classifier, 'true_positive_init'):
            return self.classifier.true_positive_init()
        else:
            return np.nan

    def error_rate_init(self):
        if hasattr(self.classifier, 'error_rate_init'):
            return self.classifier.error_rate_init()
        else:
            return np.nan

    def true_positive_update(self):
        summary = self.load_summary()
        # count of everything whose true label is new_label
        num_total = sum(sum(true_classes.values()) for true_classes in summary.values())
        # count of everything whose true and predicted label is both new_label
        num_correct = sum(
            summary[cls].get(cls, 0)
            for cls in summary.keys()
        )
        return num_correct / num_total

    def label_true_positive_update(self, new_label):
        summary = self.load_summary()
        # count of everything whose true label is new_label
        num_positive = sum(true_labels.get(new_label, 0) for true_labels in summary.values())
        # count of everything whose true and predicted label is both new_label
        num_correct = summary[new_label].get(new_label, 0)
        if num_positive == 0:
            return 1 # 100% true positive rate (over am empty dataset)
        else:
            return num_correct / num_positive

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

    def label_mean_regret_scaled(self, new_label):
        if self.label_max_regret(new_label) == 0:
            return 0
        else:
            return self.label_mean_regret(new_label) / self.label_max_regret(new_label)

    def mean_regret_scaled(self):
        """Calculate the mean scaled regret for all new labels that have data."""
        return mean(
            self.label_mean_regret_scaled(new_label)
            for new_label in self.new_ys()
        )

    def mean_regret(self):
        """Calculate the mean regret for all new labels that have data."""
        return mean(
            self.label_mean_regret(new_label)
            for new_label in self.new_ys()
        )

    def get_summary_file(self):
        return join_path(self.get_data_path(), self.get_persistent_id() + '.summary')

    def load_summary(self):
        """Load, or calculate and cache, a summary of this trial.

        Returns:
            Dict[y_hat, Dict[y, int]]: a summary of the classification distribution.
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
                for y_hat in sorted(self.old_ys()):
                    counter = stats.get(y_hat, {})
                    fd.write('    {}: {},\n'.format(repr(y_hat), repr(counter)))
                fd.write('}\n')
        try:
            with open(self.get_summary_file()) as fd:
                cached = literal_eval(fd.read())
        except SyntaxError:
            rm(self.get_summary_file())
            cached = self.load_summary()
        return cached

    def get_predictions_file(self):
        return join_path(self.get_data_path(), self.get_persistent_id() + '.predictions')

    def load_predictions(self):
        """Load, or calculate and cache, the classifications of this dataset.

        Returns:
            List[y_hat]: a summary of the classification distribution.
        """
        if not file_exists(self.get_predictions_file()):
            predictions = self.classifier.classify(self.dataset.get_x())
            with open(self.get_predictions_file(), 'w') as fd:
                fd.write('[\n')
                for y in predictions:
                    fd.write('    {},\n'.format(repr(y)))
                fd.write(']\n')
        try:
            with open(self.get_predictions_file()) as fd:
                cached = literal_eval(fd.read())
        except SyntaxError:
            rm(self.get_predictions_file())
            cached = self.load_predictions()
        return cached


def bucket_alist(alist):
    """Group keys in an association list by their values.

    Arguments:
        alist [[str, int]]: A list of [str, int] association pairs.

    Returns:
        List[(int, List[str])]: A list of grouped keys by their values.
    """
    values = dict(alist)
    keys = sorted(values.keys(), key=(lambda k: values[k]))
    bucketed = list((v, list(ks)) for v, ks in groupby(keys, key=(lambda k: values[k])))
    return bucketed


def intra_bucket_regret(bucket, groundtruth):
    """Calculate regret of one bucket.

    Arguments:
        bucket [str]: A list of labels.
        groundtruth {str:int}: Label to misclassifications dictionary.

    Returns:
        float: regret of the labels in the bucket.
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
        bucket_1 [str]: Labels in the first bucket.
        bucket_2 [str]: Labels in the second bucket.
        groundtruth {str:int}: Label to misclassifications dictionary.

    Returns:
        float: regret of the labels due to between-bucket scores.
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
        float: the mean regret over all total orders of the heuristic ranking
    """
    groundtruth = dict(groundtruth_ranking)
    heuristic_bucketed = bucket_alist(heuristic_ranking)
    regret = 0
    for i, bucket_1 in enumerate(heuristic_bucketed):
        regret += intra_bucket_regret(bucket_1[1], groundtruth)
        for bucket_2 in heuristic_bucketed[i + 1:]:
            regret += inter_bucket_regret(bucket_1[1], bucket_2[1], groundtruth)
    return regret


def bucket_to_dict(bucketed_list):
    result = {}
    for i, (_, classes) in enumerate(sorted(bucketed_list)):
        for cls in classes:
            result[cls] = i
    return result

OldLabelData = namedtuple('OldLabelData', [
    'old_class',
    'old_label',
    'old_label_size',
    'label_true_positive_init',
    'label_true_positive_update',
])

TRIAL_COLUMNS = [
    # trial information
    'trial_id',
    'mean_regret',
    'mean_regret_scaled',
    'true_positive_init',
    'true_positive_update',
    # old label information
    'old_class',
    'old_label',
    'old_label_size',
    'label_true_positive_init',
    'label_true_positive_update',
    # new label information
    'new_class',
    'new_label',
    'label_mean_regret',
    'label_max_regret',
    'label_mean_regret_scaled',
    # ranking information
    'heuristic',
    'heuristic_rank',
    'misclassification',
    'misclassification_rank',
]

TrialRow = namedtuple('TrialRow', TRIAL_COLUMNS)


def trial_to_dataframe(trial):
    # trial information
    trial_id = trial.get_persistent_id()
    mean_regret = trial.mean_regret()
    mean_regret_scaled = trial.mean_regret_scaled()
    true_positive_init = trial.true_positive_init()
    true_positive_update = trial.true_positive_update()
    # pre-calculate duplicated old label information
    old_label_data = {}
    for old_class in trial.old_ys():
        old_label_data[old_class] = OldLabelData(
            old_class=old_class,
            old_label=trial.domain_utils.class_to_label(old_class),
            old_label_size=sum(trial.load_summary()[old_class].values()),
            label_true_positive_init=trial.label_true_positive_init(old_class),
            label_true_positive_update=trial.label_true_positive_update(old_class),
        )
    # loop through new labels and create dataframe rows
    trial_df_rows = []
    for new_class in trial.new_ys():
        # new class information
        new_class = new_class
        new_label = trial.domain_utils.class_to_label(new_class)
        label_mean_regret = trial.label_mean_regret(new_class)
        label_max_regret = trial.label_max_regret(new_class)
        label_mean_regret_scaled = trial.label_mean_regret_scaled(new_class)
        # ranking information
        misclassifications = dict(trial.label_misclassification_order(new_class))
        heuristics = dict(trial.label_distance_order(new_class))
        misclassification_ranks = bucket_to_dict(bucket_alist(misclassifications))
        heuristic_ranks = bucket_to_dict(bucket_alist(heuristics))
        # cross old labels and calculate heuristic information
        for old_class in old_label_data:
            trial_df_rows.append(TrialRow(
                # trial information
                trial_id=trial_id,
                mean_regret=mean_regret,
                mean_regret_scaled=mean_regret_scaled,
                true_positive_init=true_positive_init,
                true_positive_update=true_positive_update,
                # old label information
                old_class=old_class,
                old_label=old_label_data[old_class].old_label,
                old_label_size=old_label_data[old_class].old_label_size,
                label_true_positive_init=old_label_data[old_class].label_true_positive_init,
                label_true_positive_update=old_label_data[old_class].label_true_positive_update,
                # new label information
                new_class=new_class,
                new_label=new_label,
                label_mean_regret=label_mean_regret,
                label_max_regret=label_max_regret,
                label_mean_regret_scaled=label_mean_regret_scaled,
                # ranking information
                heuristic_rank=heuristic_ranks[old_class],
                heuristic=heuristics[old_class],
                misclassification=misclassifications[old_class],
                misclassification_rank=misclassification_ranks[old_class],
            ))
    return pd.DataFrame(trial_df_rows, columns=TRIAL_COLUMNS)


def create_tribulation(directory):
    """Convert a collection of trials (a "tribulation") to a dataframe.

    Since what we are interested in is how the new classes are distributed
    among the old classes, each trial will contribute:

        |old_labels| * |new_labels|

    rows to the final dataframe. To make plotting easier, some information will
    be duplicated amongst these rows. In addition to trial-level data (which
    will be the same for all rows from a trial, information specific to the old
    label (label true positive rate at both the initialization and update
    stages) and to the new label (the mean and max regret) will also be
    duplicated.
    """
    directory = realpath(expanduser(directory))
    tribulation_file = join_path(directory, basename(directory) + '.tribulation')
    if file_exists(tribulation_file):
        return pd.read_csv(tribulation_file)
    if basename(directory).startswith('color'):
        from_file = color_from_file
    elif basename(directory).startswith('cifar'):
        from_file = image_from_file
    else:
        raise ValueError('Cannot determine domain for directory {}'.format(directory))
    trial_dfs = {}
    trials_list = [
        filename for filename in listdir(directory)
        if len(splitext(filename)[0].split('_')) == 2 and filename.endswith('.summary')
    ]
    for i, filename in enumerate(trials_list):
        trial_id = splitext(filename)[0]
        if trial_id in trial_dfs:
            continue
        print(i, len(trials_list), datetime.now().isoformat(), trial_id)
        trial_df_file = join_path(directory, trial_id + '.trial_df')
        if file_exists(trial_df_file):
            trial_df = pd.read_csv(trial_df_file)
        else:
            trial = from_file(join_path(directory, filename))
            trial_df = trial_to_dataframe(trial)
            trial_df.to_csv(trial_df_file, index=False)
        trial_dfs[trial_id] = trial_df
    tribulation_df = pd.concat(trial_dfs.values())
    tribulation_df.to_csv(tribulation_file, index=False)
    return tribulation_df


def create_color_tribulation(directory):
    df = create_tribulation(directory)
    regex = 'colors(?P<num_centroids>[0-9]*)_'
    regex += 's(?P<random_seed>[0-9.]*)'
    regex += 'n(?P<dataset_size>[0-9]*)'
    regex += 'k(?P<num_colors>[0-9]*)'
    df['regex'] = df['trial_id'].apply(
        lambda s: re.match(regex, s)
    )
    for attr in ['random_seed', 'num_centroids', 'dataset_size', 'num_colors']:
        df[attr] = df['regex'].apply(lambda match, attr=attr: match.group(attr))
        if attr != 'random_seed':
            df[attr] = df[attr].astype(int)
    del df['regex']
    return df


def create_image_tribulation(directory):
    df = create_tribulation(directory)
    df['neural_network'] = df['trial_id'].apply(
        lambda s: NeuralNetwork(join_path(directory, s.split('_')[0] + '.hdf5'))
    )
    df['int_labels'] = df['neural_network'].apply(lambda nn: tuple(nn.int_labels))
    df['batch_size'] = df['neural_network'].apply(lambda nn: nn.batch_size)
    df['num_epochs'] = df['neural_network'].apply(lambda nn: nn.num_epochs)
    del df['neural_network']
    return df
