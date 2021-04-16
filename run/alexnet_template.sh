#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=15000
#SBATCH -t 24:00:00
#SBATCH --partition=default_gpu
#SBATCH --gres=gpu:1
#SBATCH -o results_alexnet/outputs/%j.out
#SBATCH -e results_alexnet/outputs/%j.err

source activate TopNetwork

python main.py --save_dir results_alexnet/{$DATASET}_{$LR}_{$BATCH}_{$SEED} --dataset {$DATASET} --model alexnet --lr {$LR} --batch_size_train {$BATCH} --seed {$SEED}