import os
import subprocess
import itertools
import time

base_path = 'results_vgg'

if not os.path.exists(base_path):
    os.makedirs(base_path)
if not os.path.exists(base_path + "/outputs"):
    os.makedirs(base_path + "/outputs")

# experimental setup
seeds = list(range(3))
dataset = ['cifar10', 'cifar100']
learning_rates = [0.1, 0.15, 0.2]
batch_size = [32, 64, 128]
num_layers = [11, 13, 16]

grid = itertools.product(seeds, dataset, learning_rates, batch_size, num_layers)

processes = []
for s, d, lr, bs, nl in grid:
    save_dir = base_path + f"/{d}_{nl}_{lr}_{bs}_{s}"
    if os.path.exists(save_dir):
        print(f'folder already exists for {d, nl, lr, bs, s}')
        continue

    cmd = f"sbatch --requeue --export=ALL,DATASET={d},LR={lr},BATCH={bs},SEED={s},NUMLAYERS={nl} run/vgg/template.sh"

    print(cmd)

    f = open(save_dir + '.log', 'w')
    subprocess.Popen(cmd.split(), stdout=f, stderr=f)
    