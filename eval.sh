#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=1:00:00
#SBATCH --output=logs/resent50_mlp_1_2_eval.out

# python train_prohmr_surfnormals_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/16489/best_global_model.pt --log_step 250 
python eval_regression_fusion_egobody.py --checkpoint /home/kotaik/DH/EgoDepth-HMR/tmp/19063/best_global_model.pt --dataset_root /work/courses/digital_human/13/egobody_release
