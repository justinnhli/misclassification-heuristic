#!/bin/bash

clusterun.py --labels="$(./powerset.py)" './run-cifar10-all.py $labels'
