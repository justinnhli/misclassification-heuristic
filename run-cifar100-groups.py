#!/usr/bin/env python3

import sys
import re
from itertools import combinations
from random import Random

from clusterun import run_cli


def list_random_seeds():
    rand = Random()
    rand.seed(8675309)
    return [rand.random() for _ in range(40)]


def generate_jobs():
    int_labels = list_random_seeds()
    breaks = list(range(0, len(int_labels), len(int_labels) // 40)) + [len(int_labels)]
    ranges = list(zip(breaks[:-1], breaks[1:]))
    job_name = 'cifar10-10C3-history'
    variables = [
        (
            'ranges',
            [f'{start}-{end}' for start, end in ranges],
        ),
    ]
    commands = [
        'cd /home/justinnhli/git/misclassification-heuristic',
        ' '.join([
            '/home/justinnhli/glibc-install/lib/ld-linux-x86-64.so.2',
            '--library-path /home/justinnhli/glibc-install/lib:/home/justinnhli/gcc-install/lib64/:/lib64:/usr/lib64',
            '/home/justinnhli/.venv/misclassification-heuristic/bin/python3',
            r'run-cifar10-all.py "$ranges"',
        ]),
    ]
    run_cli(job_name, variables, commands, venv='misclassification-heuristic')


def run_job(start, end):
    from images import train_neural_network
    from images import ImageUtils, ImageDataset, NeuralNetwork
    from classifiers import RegretTrial
    for seed in list_random_seeds()[start:end]:
        rand = Random()
        rand.seed(seed)
        training_labels = [
            *sample(range(40, 45), 3),
            *sample(range(60, 65), 3),
            *sample(range(65, 70), 3),
        ]
        network_files = train_neural_network(
            training_labels,
            batch_size=32,
            num_epochs=200,
            dataset_str='cifar100',
            output_path='cifar100-groups',
            verbose=False,
        )
        for network_file in network_files:
            network = NeuralNetwork(network_file)
            trial = RegretTrial(
                network,
                ImageUtils('cifar100'),
                ImageDataset('cifar100'),
                path_prefix=directory,
            )
            trial.load_summary()


def main():
    error = False
    if len(sys.argv) == 2:
        if sys.argv[1] == '--generate-jobs':
            generate_jobs()
        elif re.match('^[0-9]+-[0-9]+$', sys.argv[1]):
            run_job(*(int(i) for i in sys.argv[1].split('-')))
        else:
            error = True
    else:
        error = True
    if error:
        print('Unrecognized arguments: ' + ' '.join(sys.argv))
        print(f'Usage: {sys.argv[0]} ( --generate-jobs | <index> )')
        exit(1)


if __name__ == '__main__':
    main()
