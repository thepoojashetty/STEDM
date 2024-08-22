#!/bin/bash -l
#SBATCH --job-name=stylediff
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff_train_ssl.out
#SBATCH -e stylediff_train_ssl.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

srun python train_simCLR.py location=cluster  data=cityscapes data/dataset=[cityscapes_simclr] style_sampling=augmented style_agg=mean