#!/usr/bin/env python3

import sys
import re
from itertools import combinations

from clusterun import run_cli
from images import train_neural_network
from images import ImageUtils, ImageDataset
from classifiers import RegretTrial


def generate_jobs():
    job_name = 'cifar10-10C3-history'
    variables = [
        (
            'int_labels',
            [
                ''.join(str(n) for n in subset)
                for subset in combinations(range(10), 3)
            ],
        ),
    ]
    commands = [
        'cd /home/justinnhli/git/misclassification-heuristic',
        ' '.join([
            '/home/justinnhli/glibc-install/lib/ld-linux-x86-64.so.2',
            '--library-path /home/justinnhli/glibc-install/lib:/home/justinnhli/gcc-install/lib64/:/lib64:/usr/lib64',
            '/home/justinnhli/.venv/misclassification-heuristic/bin/python3',
            r'run-cifar10-all.py "$int_labels"',
        ]),
    ]
    run_cli(job_name, variables, commands, venv='misclassification-heuristic')


def run_job(int_labels):
    directory = 'cifar10-threes-history-new'
    nn = train_neural_network(
        int_labels,
        batch_size=32,
        num_epochs=200,
        dataset_str='cifar10',
        output_path=directory,
        verbose=False,
        checkpoint=True,
    )
    RegretTrial(
        nn,
        ImageUtils('cifar10'),
        ImageDataset('cifar10'),
        path_prefix=directory,
    )


def main():
    error = False
    if len(sys.argv) == 2:
        if sys.argv[1] == '--generate-jobs':
            generate_jobs()
        elif re.match('^[0-9]{3}', sys.argv[1]):
            run_job([int(c) for c in sys.argv[1]])
        else:
            error = True
    else:
        error = True
    if error:
        print('Unrecognized arguments: ' + ' '.join(sys.argv))
        print(f'Usage: {sys.argv[0]} ( --generate-jobs | [0-9]{{3}} )')
        exit(1)


if __name__ == '__main__':
    main()
