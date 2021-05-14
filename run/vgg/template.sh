#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=128000
#SBATCH -t 48:00:00
#SBATCH --partition=cuvl
#SBATCH --gres=gpu:1
#SBATCH -o results_vgg/outputs/%j.out
#SBATCH -e results_vgg/outputs/%j.err

source activate TopNetwork

python main.py --save_file dims_vgg.txt --meta_data ${DATASET}_${NUMLAYERS}_${LR}_${BATCH}_${SEED} --dataset $DATASET --model vgg --lr $LR --batch_size_train $BATCH --seed $SEED --depth $NUMLAYERS