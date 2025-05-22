#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=6:00:00
#SBATCH --output=logs/resent50_mlp_1_2.out

# python train_prohmr_surfnormals_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/16489/best_global_model.pt --log_step 250 

# python train_prohmr_fusion_egobody.py --load_depth_pretrained true --load_only_backbone true --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --log_step 250 
python train_prohmr_fusion_egobody.py --load_depth_pretrained true --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --load_pretrained true --checkpoint /home/kotaik/DH/EgoDepth-HMR/tmp/27457/best_global_model.pt --log_step 250 
