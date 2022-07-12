#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 4 
#SBATCH --mem-per-cpu=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -C gmem12
#SBATCH -o logs/eval/output_eval_%j.out

module load anaconda3

python process_gallery_C.py

python process_videos_C.py

python evaluationC.py 

