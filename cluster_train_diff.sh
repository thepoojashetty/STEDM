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

srun python train_diff.py location=cluster  data=flowers data/dataset=[flowers_anno] style_sampling=augmented style_agg=mean data.class_train_samples=3