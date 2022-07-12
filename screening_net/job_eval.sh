#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 8 
#SBATCH --mem-per-cpu=8G

#SBATCH -o logs/eval/output_eval_epoch150_finetuned_%j.out

module load anaconda3

#python process_gallery_C.py

#python process_videos_C.py

python evaluationC_save.py 
#python evaluationC.py


