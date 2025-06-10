#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=1:00:00
#SBATCH --output=logs/resent50_fusion_attn_eval_test.out


python eval_regression_fusion_egobody.py \
    --checkpoint model_checkpoints/fusion_attn/best_model.pt \
    --data_root /work/courses/digital_human/13