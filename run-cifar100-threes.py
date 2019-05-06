#!/usr/bin/env python3

import sys
import re
from itertools import combinations
from random import Random

from clusterun import run_cli

from trial_dataframe import trial_to_dataframe


def list_int_labels():
    rng = Random(8675309)
    results = set()
    for _ in range(120):
        numbers = set()
        while len(numbers) != 3 or tuple(sorted(numbers)) in results:
            numbers = set([rng.randrange(100), rng.randrange(100), rng.randrange(100)])
        results.add(tuple(sorted(numbers)))
    return sorte(results)


def generate_jobs():
    int_labels = list_int_labels()
    breaks = list(range(0, len(int_labels), len(int_labels) // 40)) + [len(int_labels)]
    ranges = list(zip(breaks[:-1], breaks[1:]))
    job_name = 'cifar100-threes'
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
            r'run-cifar100-threes.py "$ranges"',
        ]),
    ]
    run_cli(job_name, variables, commands, venv='misclassification-heuristic')


def run_job(start, end):
    from images import train_neural_network
    from images import ImageUtils, ImageDataset, NeuralNetwork
    from classifiers import RegretTrial
    for int_labels in list_int_labels()[start:end]:
        directory = 'cifar100-threes'
        network_files = train_neural_network(
            int_labels,
            batch_size=32,
            num_epochs=200,
            dataset_str='cifar10',
            output_path=directory,
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
            trial_to_dataframe(trial)


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
