#!/bin/bash -l
#SBATCH --job-name=msscc_uncdiff_mp_fm
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH -o msscc_uncdiff_mp_fm.out
#SBATCH -e msscc_uncdiff_mp_fm.err
#SBATCH --time=2:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

#Cityscapes

# srun python feature_matching.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] \
#     style_sampling=augmented style_agg=mean \
#     cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune_last.ckpt \
#     +run_name=cityscapes_unconddiff_fm

# srun python feature_matching.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=SimCLR_cityscapes_sameaug_finetune_last.ckpt \
#      encoder=simclr +encoder_ckpt=SimCLR_cityscapes_sameaug_last.ckpt \
#      +run_name=cityscapes_simclr_fm

#MSSCC

#nearby
# srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune_last-v1.ckpt \
#      +run_name=msscc_uncdiff_nearby_fm

# srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +run_name=msscc_simclr_nearby_fm

# srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +run_name=msscc_simclr2_nearby_fm

# mp linear
srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
     cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_MP_Linear_MSSCC_finetune_last-v1.ckpt \
     +run_name=msscc_uncdiff_mp_fm

# srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +run_name=msscc_simclr_mp_fm

# srun python feature_matching.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +run_name=msscc_simclr2_mp_fm