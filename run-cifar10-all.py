#!/usr/bin/env python3

import sys
from itertools import chain, combinations

from images import train_neural_network

def main():
    int_labels = [int(c) for c in sys.argv[1]]
    train_neural_network(
        int_labels,
        batch_size=32,
        num_epochs=2,
        dataset_str='cifar10',
        output_path='cifar10-all',
        verbose=False,
        checkpoint=True,
    )

if __name__ == '__main__':
    main()
