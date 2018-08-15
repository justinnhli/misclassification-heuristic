import re
from collections import namedtuple
from datetime import datetime
from os import listdir
from os.path import join as join_path, exists as file_exists, realpath, expanduser, basename, splitext

import pandas as pd

from classifiers import bucket_alist
from colors import from_file as color_from_file
from images import from_file as image_from_file, NeuralNetwork

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
                heuristic=heuristic_ranks[old_class],
                heuristic_rank=heuristics[old_class],
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
    df['int_labels'] = df['neural_network'].apply(lambda nn: nn.int_labels)
    df['batch_size'] = df['neural_network'].apply(lambda nn: nn.batch_size)
    df['num_epochs'] = df['neural_network'].apply(lambda nn: nn.num_epochs)
    del df['neural_network']
    return df
