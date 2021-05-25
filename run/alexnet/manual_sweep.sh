#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=35000
#SBATCH -t 48:00:00
#SBATCH --partition=orion
#SBATCH --gres=gpu:1
#SBATCH -o null
#SBATCH -e null

source activate TopNetwork

python manual_sweep 0 4