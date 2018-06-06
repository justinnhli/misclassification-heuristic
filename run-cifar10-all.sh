#!/bin/bash

read -r -d '' commands << EOM
source "/home/justinnhli/.venv/misclassification-heuristic/bin/activate"
./run-cifar10-all.py \$labels
deactivate
EOM
clusterun.py --labels="$(./powerset.py)" "$commands"
