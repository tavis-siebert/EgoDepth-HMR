#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=10:00:00
#SBATCH --output=logs/resent50_fusion_continue5.out

python train_prohmr_fusion_egobody.py --load_depth_pretrained true --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --log_step 100 --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/84608/best_global_model.pt
