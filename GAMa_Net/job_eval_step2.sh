#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH -o logs/eval/final_eval_8seq2dcnn_fepoch180_Satepoch400_top10per_%j.out

module load anaconda3

#python process_gallery_C.py

#python process_videos_C.py

python evaluationC_pred_Laerial_test.py 

