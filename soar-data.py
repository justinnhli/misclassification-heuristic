import pandas as pd
import numpy as np

COLUMNS = [
    'num_objects',
    'num_train',
    'num_test',
    'memory',
    'strategy',
    'seed',
    'depth',
    'num_diff',
    'status',
    'guesses',
]

df = (
        pd
        .read_csv('soar-data.tsv', sep='\t', names=COLUMNS)
        .pivot_table(
            index=['memory', 'strategy'],
            columns=['num_diff', 'status'],
            values='guesses',
            aggfunc=['count', np.mean, np.std]
        ).reorder_levels([1, 2, 0], axis=1)
        .sort_index(axis=1)
        .fillna('-')
)
print(df)
