#!/bin/bash -l
#SBATCH --job-name=stylediff
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff.out
#SBATCH -e stylediff.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

RESULTS_SUB_FOLDER="cityscapes_no_vert_hor_aug"
# CKPT_NAME="Diff_flowers_3_augmented_last.ckpt"
# CKPT_NAME="Diff_cityscapes_20_augmented_last.ckpt"
CKPT_NAME="Diff_cityscapes_augmented_50_epochs_last.ckpt"

srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean cfg_scale=1.5 +ckpt_name=$CKPT_NAME +predict_dir=$RESULTS_SUB_FOLDER