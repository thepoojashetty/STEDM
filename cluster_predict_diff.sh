#!/bin/bash -l
#SBATCH --job-name=stylediff_pred_uncdiff_c
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH -o stylediff_pred_uncdiff_c.out
#SBATCH -e stylediff_pred_uncdiff_c.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_SSL_Augmented_Mean_Cityscapes_vanilla_last.ckpt \
#      +predict_dir=Predict_Diff_SSL_Augmented_Mean_Cityscapes_vanilla

# Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune
# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune_last.ckpt \
#      +predict_dir=Predict_Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune

# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=SimCLR_cityscapes_finetune_last.ckpt \
#      +predict_dir=Predict_SimCLR_cityscapes_finetune encoder=simclr +encoder_ckpt=SimCLR_cityscapes_last.ckpt

# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=SimCLR_cityscapes_sameaug_finetune_last.ckpt \
#      +predict_dir=Predict_SimCLR_cityscapes_sameaug_finetune encoder=simclr +encoder_ckpt=SimCLR_cityscapes_sameaug_last.ckpt

#################################################
#MSSCC
#vanilla
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Nearby_Mean_MSSCC_vanilla_last-v2.ckpt \
#      +predict_dir=Predict_Diff_Nearby_Mean_MSSCC_vanilla

# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune_last-v1.ckpt \
#      +predict_dir=Predict_Diff_Uncdiff_Nearby_Mean_MSSCC_finetune

# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +predict_dir=Predict_SimCLR_MSSCC_Nearby_finetune

#################################################
#linear for mp
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      cfg_scale=1.5 +ckpt_name=Diff_MP_Linear_MSSCC_vanilla_last-v1.ckpt \
#      +predict_dir=Predict_Diff_MP_Linear_MSSCC_vanilla

# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +predict_dir=Predict_SimCLR_MSSCC_MP_linear_finetune

#unconditional mp linear msscc
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_MP_Linear_MSSCC_finetune_last-v1.ckpt \
#      +predict_dir=Predict_Diff_Uncdiff_MP_Linear_MSSCC_finetune data.samples=2000

####################################################

#msscc: simclr2: nearby
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +predict_dir=Predict_SimCLR2_MSSCC_Nearby_finetune data.samples=1300

#msscc: simclr2: mp: linear
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +predict_dir=Predict_SimCLR2_MSSCC_MP_linear_finetune data.samples=1300

#################################################
#prediction with feature matching
####################################################

#cityscapes : uncondiff
# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune_last.ckpt \
#      +predict_dir=Predict_fm_Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune data.samples=416 \
#      fm=true

#cityscape : simclr
# srun python predict_diff.py location=cluster data=cityscapes data/dataset=[cityscapes_anno,cityscapes_unanno] data.ratios=[1.0,1.0] style_sampling=augmented style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=SimCLR_cityscapes_sameaug_finetune_last.ckpt \
#      +predict_dir=Predict_fm_SimCLR_cityscapes_sameaug_finetune \
#      encoder=simclr +encoder_ckpt=SimCLR_cityscapes_sameaug_last.ckpt \
#      fm=true data.samples=464

#msscc : uncondiff : nearby
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_Diff_Uncdiff_Nearby_Mean_MSSCC_finetune data.samples=1300 \
#      fm=true

#msscc : uncondiff : mp: linear
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      cfg_scale=1.5 +ckpt_name=Diff_Uncdiff_MP_Linear_MSSCC_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_Diff_Uncdiff_MP_Linear_MSSCC_finetune data.samples=1300\
#      fm=true

#msscc : simclr : nearby
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_SimCLR_MSSCC_Nearby_finetune data.samples=1300 \
#      fm=true

#msscc : simclr : mp: linear
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR_msscc_last-v3.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_SimCLR_MSSCC_MP_linear_finetune data.samples=1300 \
#      fm=true

#msscc : simclr2 : nearby
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=nearby style_agg=mean \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_Nearby_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_SimCLR2_MSSCC_Nearby_finetune data.samples=1300 \
#      fm=true

#msscc : simclr2 : mp: linear
# srun python predict_diff.py location=cluster data=msscc data/dataset=[msscc_anno,msscc_unanno] data.ratios=[1.0,1.0] style_sampling=mp style_agg=linear \
#      encoder=simclr +encoder_ckpt=SimCLR2_msscc_last-v2.ckpt \
#      cfg_scale=1.5 +ckpt_name=SimCLR2_MSSCC_MP_linear_finetune_last-v1.ckpt \
#      +predict_dir=Predict_fm_SimCLR2_MSSCC_MP_linear_finetune data.samples=1300 \
#      fm=true
