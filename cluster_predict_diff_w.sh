#!/bin/bash -l
#SBATCH --job-name=stylediff
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o stylediff_pred.out
#SBATCH -e stylediff_pred.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# CKPT_NAME="Diff_flowers_3_augmented_last.ckpt"
# CKPT_NAME="Diff_cityscapes_20_augmented_last.ckpt"
# CKPT_NAME="Diff_cityscapes_augmented_50_epochs_last.ckpt"
CKPT_NAME="Diff_cityscapes_augmented_ssl_last.ckpt"
# cfg_scale=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7,8,9]
srun python predict_diff_w.py location=cluster data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean cfg_scale=[1.5] +ckpt_name=$CKPT_NAME