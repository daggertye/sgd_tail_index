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

def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]

def calculate_ph_dim(W, min_points=200, max_points=1000, point_jump=50, h_dim=0, 
                     print_error=False):
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        diagrams = ripser(sample_W(W, n))['dgms']
        
        if len(diagrams) > h_dim:
            d = diagrams[h_dim]
            d = d[d[:, 1] < np.inf]
            lengths.append((d[:, 1] - d[:, 0]).sum())
        else:
            lengths.append(0.0)
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="results_alexnet", type=str)
    parser.add_argument('--save_file', default="dims_alexnet.txt", type=str)
    args = parser.parse_args()


    if not os.path.exists(args.model_dir):
        print(f"Model Directory {args.model_dir} does not exist!")
        exit()

    output_file = open(args.save_file, 'w')

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
        ph_dim = calculate_ph_dim(weights)

        #write
        test_history = torch.load(os.path.join(args.model_dir, f, 'evaluation_history_TEST.hist'))
        test_history = torch.tensor(test_history)
        output_file.write(f"Run: {f}, train acc: {eval_history[-1, -1]:.2f}, test acc: {test_history[-1, -1]:.2f}, ph_dim: {ph_dim:.3f}\n")
    output_file.close()