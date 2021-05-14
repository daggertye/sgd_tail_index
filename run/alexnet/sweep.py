import os
import subprocess
import itertools
import time

# experimental setup
seeds = list(range(3))
optimizers = ['Adam', 'RMSprop']
dataset = ['cifar10', 'cifar100']
learning_rates = [0.01, 0.05, 0.1]
batch_size = [128, 192, 256]

grid = itertools.product(seeds, optimizers, dataset, learning_rates, batch_size)

processes = []
for s, o, d, lr, bs in grid:
    cmd = f"sbatch --requeue --export=ALL,DATASET={d},LR={lr},BATCH={bs},SEED={s},OPTIM={o} run/alexnet/template.sh"

    print(cmd)

    os.system(cmd)
    