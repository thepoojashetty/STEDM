#!/bin/bash -l
#SBATCH --job-name=unconditional_ssl_m
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=a40
#SBATCH -o stylediff_train_ssl_m.out
#SBATCH -e stylediff_train_ssl_m.err
#SBATCH --time=24:00:00

module load python
conda activate stylediff

export https_proxy="http://proxy.rrze.uni-erlangen.de:80"

# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_ssl] style_sampling=augmented style_agg=mean \
#     +run_name=Diff_Uncdiff_Augmented_Mean_Cityscapes \
#     training_type=ssl

# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_anno] style_sampling=augmented style_agg=mean \
#     +ckpt_path=Diff_Uncdiff_Augmented_Mean_Cityscapes_last.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_Augmented_Mean_Cityscapes_finetune

# srun python train_diff.py location=cluster  data=cityscapes data/dataset=[cityscapes_ssl] style_sampling=augmented style_agg=mean \
#     +run_name=Diff_Uncdiff_Augmented_Mean_Cityscapes_rain_only \
#     training_type=ssl

#####################################################################################

# nearby unconditional ssl training
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=nearby style_agg=mean \
#     +run_name=Diff_Uncdiff_Nearby_Mean_MSSCC \
#     training_type=ssl

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=nearby style_agg=mean \
#     +run_name=Diff_Uncdiff_Nearby_Mean_MSSCC +ckpt_name=Diff_Uncdiff_Nearby_Mean_MSSCC_last-v1.ckpt \
#     training_type=ssl

#FINETUNE
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     +ckpt_path=Diff_Uncdiff_Nearby_Mean_MSSCC_last-v2.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=nearby style_agg=mean \
#     +ckpt_path=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune_last.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_Nearby_Mean_MSSCC_finetune

#mp unconditional ssl training
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=mp style_agg=mean \
#     +run_name=Diff_Uncdiff_MP_Mean_MSSCC location.batch_mul=2\
#     training_type=ssl

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=mp style_agg=mean \
#     +run_name=Diff_Uncdiff_MP_Mean_MSSCC location.batch_mul=2 +ckpt_name=Diff_Uncdiff_MP_Mean_MSSCC_last-v1.ckpt\
#     training_type=ssl

#FINETUNE
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#     +ckpt_path=Diff_Uncdiff_MP_Mean_MSSCC_last-v2.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_MP_Mean_MSSCC_finetune

# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=mean \
#     +ckpt_path=Diff_Uncdiff_MP_Mean_MSSCC_finetune_last.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_MP_Mean_MSSCC_finetune

#########################################
#mp linear
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=mp style_agg=linear \
#     +run_name=Diff_Uncdiff_MP_Linear_MSSCC location.batch_mul=2\
#     training_type=ssl

#continue
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_ssl] style_sampling=mp style_agg=linear \
#     +run_name=Diff_Uncdiff_MP_Linear_MSSCC location.batch_mul=2 +ckpt_path=Diff_Uncdiff_MP_Linear_MSSCC_last.ckpt \
#     training_type=ssl num_epochs=7

#FINETUNE
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     +ckpt_path=Diff_Uncdiff_MP_Linear_MSSCC_last-v1.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_MP_Linear_MSSCC_finetune

#continue
# srun python train_diff.py location=cluster  data=msscc data/dataset=[msscc_anno] style_sampling=mp style_agg=linear \
#     +ckpt_path=Diff_Uncdiff_MP_Linear_MSSCC_finetune_last.ckpt \
#     training_type=finetune \
#     +run_name=Diff_Uncdiff_MP_Linear_MSSCC_finetune num_epochs=7
