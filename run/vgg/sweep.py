import os
import subprocess
import itertools
import time

# experimental setup
seeds = list(range(3))
dataset = ['cifar10', 'cifar100']
learning_rates = [0.01, 0.05, 0.1]
batch_size = [128, 192, 256]
num_layers = [11, 13, 16]

grid = itertools.product(seeds, dataset, learning_rates, batch_size, num_layers)

processes = []
for s, d, lr, bs, nl in grid:
    cmd = f"sbatch --requeue --export=ALL,DATASET={d},LR={lr},BATCH={bs},SEED={s},NUMLAYERS={nl} run/vgg/template.sh"

    print(cmd)

    os.system(cmd)
    