#!/bin/bash -l
#SBATCH --job-name=stylediff
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff_train.out
#SBATCH -e stylediff_train.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

#vanilla training
# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean num_epochs=50

#for ssl training finetuning with unconditional diffusion
srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean +ckpt_path=Diff_SSL_Augmented_Mean_Cityscapes_last.ckpt