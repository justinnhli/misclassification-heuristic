#!/usr/bin/env python3

from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def main():
    print('[' + 
        ','.join(
            '"{}"'.format(''.join(str(n) for n in sorted(labels)))
            for labels in powerset(range(10))
            if 2 <= len(labels) <= 9
        ) + ']'
    )


if __name__ == '__main__':
    main()
