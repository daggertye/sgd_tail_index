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
learning_rates = [0.1, 0.15, 0.2]
batch_size = [32, 64, 128]

grid = itertools.product(seeds, dataset, learning_rates, batch_size)

processes = []
for s, d, lr, bs in grid:
    save_dir = base_path + f"/{d}_{lr}_{bs}_{s}"
    if os.path.exists(save_dir):
        print(f'folder already exists for {d, lr, bs, s}')
        continue

    cmd = f"sbatch --requeue --export=ALL,DATASET={d},LR={lr},BATCH={bs},SEED={s} run/alexnet_template.sh"

    print(cmd)

    f = open(save_dir + '.log', 'w')
    subprocess.Popen(cmd.split(), stdout=f, stderr=f)
    