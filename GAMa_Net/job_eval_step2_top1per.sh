#!/bin/bash
#SBATCH -n 1 
#SBATCH -c 5
#SBATCH --mem-per-cpu=12G
#SBATCH -o logs/eval/final_eval_32_2dcnn_Satepoch400_top1per_%j.out

module load anaconda3

#python process_gallery_C.py

#python process_videos_C.py

python evaluationC_pred_Laerial_test_top1per.py 

#final_eval_Laerial8random_Satepoch400_top1per_%j.out
#check embedding for full and finetuned
