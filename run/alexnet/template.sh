#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=35000
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o null
#SBATCH -e null

source activate TopNetwork

python main.py --save_file dims_alexnet_optimizers.txt --meta_data ${DATASET}_${OPTIM}_${LR}_${BATCH}_${SEED} --dataset $DATASET --model alexnet --lr $LR --batch_size_train $BATCH --seed $SEED