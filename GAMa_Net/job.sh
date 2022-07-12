#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 6 
#SBATCH --mem-per-cpu=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem48
#SBATCH -o logs/output_%j.out

module load anaconda3

python main.py

