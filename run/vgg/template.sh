#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=25000
#SBATCH -t 48:00:00
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:1
#SBATCH -o results_vgg/outputs/%j.out
#SBATCH -e results_vgg/outputs/%j.err

source activate TopNetwork

python main.py --save_dir results_vgg/${DATASET}_${NUMLAYERS}_${LR}_${BATCH}_${SEED} --dataset $DATASET --model vgg --lr $LR --batch_size_train $BATCH --seed $SEED --depth $NUMLAYERS