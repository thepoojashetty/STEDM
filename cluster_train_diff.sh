#!/bin/bash -l
#SBATCH --job-name=stylediff_van_m
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff_van_m.out
#SBATCH -e stylediff_van_m.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

#vanilla training
# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#      +run_name=Diff_SSL_Augmented_Mean_Cityscapes_vanilla

#training on vanilla with clear,foggy and rainy data.
# This is to verify how the style extractor works
# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#      +run_name=Diff_SSL_Augmented_Mean_Cityscapes_vanilla_all_data

# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#      +run_name=Diff_SSL_Augmented_Mean_Cityscapes_vanilla_only_rainy

#######################################
#nearby vanilla training
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#      +run_name=Diff_Nearby_Mean_MSSCC_vanilla

#continuing from last checkpoint
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#      +run_name=Diff_Nearby_Mean_MSSCC_vanilla +ckpt_name=Diff_Nearby_Mean_MSSCC_vanilla_last-v1.ckpt

#mp vanilla training
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#      +run_name=Diff_MP_Mean_MSSCC_vanilla location.batch_mul=2

#continuing from last checkpoint
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#      +run_name=Diff_MP_Mean_MSSCC_vanilla location.batch_mul=2 +ckpt_name=Diff_MP_Mean_MSSCC_vanilla_last-v1.ckpt

#########################
#mp with linear aug
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#      +run_name=Diff_MP_Linear_MSSCC_vanilla location.batch_mul=2

#continuing from last checkpoint
srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
     +run_name=Diff_MP_Linear_MSSCC_vanilla location.batch_mul=2 +ckpt_name=Diff_MP_Linear_MSSCC_vanilla_last.ckpt\
     num_epochs=7