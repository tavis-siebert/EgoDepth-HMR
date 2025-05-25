#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=8:00:00
#SBATCH --output=logs/resent50_surfnorm.out

# python train_prohmr_surfnormals_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/16489/best_global_model.pt --log_step 250 
python train_prohmr_surfnormals_egobody.py --log_step 366 --load_pretrained true --checkpoint /work/courses/digital_human/13/weiwan/surf_normals/best_global_model_surf.pt