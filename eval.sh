#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=00:10:00
#SBATCH --output=logs/resent50_surfnorm_eval_test_2.out

python eval_regression_surfnorm_egobody.py \
    --checkpoint /work/courses/digital_human/13/kotaik/surf_norm_checkpoints/44899/best_global_model.pt \
    --data_root /work/courses/digital_human/13 \
    --model_cfg prohmr/configs/prohmr.yaml
