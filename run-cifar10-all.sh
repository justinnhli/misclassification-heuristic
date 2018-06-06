#!/bin/bash

read -r -d '' commands << EOM
source "/home/justinnhli/.venv/misclassification-heuristic/bin/activate"
/home/justinnhli/glibc-install/lib/ld-linux-x86-64.so.2 --library-path /home/justinnhli/glibc-install/lib:/home/justinnhli/gcc-install/lib64/:/lib64:/usr/lib64 ~/.venv/misclassification-heuristic/bin/python3 run-cifar10-all.py \$labels
deactivate
EOM
clusterun.py --labels="$(./powerset.py)" "$commands"

# testing:
#clusterun.py --labels="'01234'" "$commands"
