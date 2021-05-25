import os
import subprocess
import itertools
import time
import sys

mod = int(sys.argv[1])
modulo = int(sys.argv[2])

assert modulo > mod

# experimental setup
datasets = ['cifar10', 'cifar100']
seeds = list(range(3))
learning_rates = [5e-3, 0.01, 0.05, 0.1]
batch_size = [64, 128, 192, 256]

grid = itertools.product(datasets, seeds, learning_rates, batch_size)

i = 0
for d, s, lr, bs, in grid:
    if i % modulo == mod:
        cmd = f"python main.py --save_file dims_alexnet.txt --meta_data {d}_{lr}_{bs}_{s} --dataset {d} --model alexnet --lr {lr} --batch_size_train {bs} --seed {s}"
        print(cmd)
        os.system(cmd)
    i += 1