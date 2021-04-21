import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import persim
import ripser
import torch

from persim import plot_diagrams
from ripser import ripser


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="results_alexnet", type=str)
    parser.add_argument('--save_dir', default="viz_alexnet", type=str)
    args = parser.parse_args()


    if not os.path.exists(args.model_dir):
        print(f"Model Directory {args.model_dir} does not exist!")
        exit()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # loop through the values
    for f in os.listdir(args.model_dir):
        if not os.path.isdir(os.path.join(args.model_dir, f)):
            continue
        if f == 'outputs':
            continue

        # make sure eval history is gucc
        eval_history = torch.load(os.path.join(args.model_dir, f, 'evaluation_history_TRAIN.hist'))
        if len(eval_history) == 0:
            continue
        eval_history = torch.tensor(eval_history)
        if eval_history[:, -1].max() < 0.8:
            continue
        if torch.isnan(eval_history).any():
            continue
        
        # check weight path for enough iterations
        weights_path = os.path.join(args.model_dir, f, 'weights')
        if len(os.listdir(weights_path)) < 1000:
            continue

        #reorder the weights since they might be out of order
        weight_iters = []
        for g in os.listdir(weights_path):
            weight_iters.append(int(g[:-3]))
        weight_iters.sort()

        # get weights from files
        weights = []
        for w in weight_iters:
            weights.append(torch.load(os.path.join(weights_path, str(w) + '.pt')))
        weights = torch.stack(weights).detach().numpy()

        # calculate ph
        diagrams = ripser(weights, maxdim=1)['dgms']
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plot_diagrams(diagrams, show=False, lifetime=True)
        plt.savefig(os.path.join(args.save_dir, f + '.png'))
        plt.close(fig)
