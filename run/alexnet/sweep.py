import os
import subprocess
import itertools
import time

base_path = 'results_alexnet'

if not os.path.exists(base_path):
    os.makedirs(base_path)
if not os.path.exists(base_path + "/outputs"):
    os.makedirs(base_path + "/outputs")

# experimental setup
seeds = list(range(3))
dataset = ['cifar10', 'cifar100']
learning_rates = [0.01, 0.05, 0.1]
batch_size = [128, 192, 256]

grid = itertools.product(seeds, dataset, learning_rates, batch_size)

processes = []
for s, d, lr, bs in grid:
    save_dir = base_path + f"/{d}_{lr}_{bs}_{s}"
    if os.path.exists(save_dir):
        print(f'folder already exists for {d, lr, bs, s}')
        continue

    cmd = f"sbatch --requeue --export=ALL,DATASET={d},LR={lr},BATCH={bs},SEED={s} run/alexnet/template.sh"

    print(cmd)

    subprocess.Popen(cmd.split())
    