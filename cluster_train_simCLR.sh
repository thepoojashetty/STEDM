#!/bin/bash -l
#SBATCH --job-name=simclr2_m
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff_train_simclr2_m.out
#SBATCH -e stylediff_train_simclr2_m.err
#SBATCH --time=24:00:00
#SBATCH --exclude=a0225,a1621,a1622

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# srun python train_simCLR.py location=cluster  data=cityscapes data/dataset=[cityscapes_simclr] style_sampling=augmented style_agg=mean

# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_cityscapes_last.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR_cityscapes_finetune

#was once trained. now contiinuing from last checkpoint
# srun python train_simCLR.py location=cluster data=msscc data/dataset=[msscc_simclr] +ckpt_name=SimCLR_msscc_last-v2.ckpt

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR_MSSCC_Nearby_finetune

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR_MSSCC_MP_finetune

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune +ckpt_name=SimCLR_MSSCC_MP_finetune_last.ckpt\
#     +run_name=SimCLR_MSSCC_MP_finetune num_epochs=6

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune +ckpt_name=SimCLR_MSSCC_Nearby_finetune_last.ckpt\
#     +run_name=SimCLR_MSSCC_Nearby_finetune num_epochs=6




#cityscapes training with same style transforms
# srun python train_simCLR.py location=cluster  data=cityscapes data/dataset=[cityscapes_simclr] \
#     style_sampling=augmented style_agg=mean\
#     +run_name=SimCLR_cityscapes_sameaug

#finetune
# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR_cityscapes_sameaug_last.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR_cityscapes_sameaug_finetune




# msscc simclr finetune with mp linear
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR_MSSCC_MP_linear_finetune

#continue
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt\
#     training_type=finetune +ckpt_name=SimCLR_MSSCC_MP_linear_finetune_last.ckpt \
#     +run_name=SimCLR_MSSCC_MP_linear_finetune num_epochs=7




#msscc, augmented views from same patient
# srun python train_simCLR.py location=cluster data=msscc data/dataset=[msscc_simclr_2] +ckpt_name=SimCLR2_msscc_last-v1.ckpt\
#     +run_name=SimCLR2_msscc

#finetune, nearby
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR2_MSSCC_Nearby_finetune

#continue
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt\
#     training_type=finetune +ckpt_name=SimCLR2_MSSCC_Nearby_finetune_last.ckpt\
#     +run_name=SimCLR2_MSSCC_Nearby_finetune num_epochs=2

#finetune, mp
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt\
#     training_type=finetune \
#     +run_name=SimCLR2_MSSCC_MP_linear_finetune

#continue
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt\
#     training_type=finetune +ckpt_name=SimCLR2_MSSCC_MP_linear_finetune_last.ckpt\
#     +run_name=SimCLR2_MSSCC_MP_linear_finetune num_epochs=6