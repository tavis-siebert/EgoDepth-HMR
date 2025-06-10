#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=8:00:00
#SBATCH --output=logs/resent50_surfnorm_2.out

# python train_prohmr_surfnormals_egobody.py --log_step 366 --load_pretrained true --checkpoint /work/courses/digital_human/13/weiwan/surf_normals/best_global_model_surf.pt
python train_prohmr_surfnormals_egobody.py \
    --data_root /work/courses/digital_human/13 \
    --log_step 366 \
    --load_pretrained true \
    --checkpoint /home/kotaik/DH/EgoDepth-HMR/tmp/75586/best_global_model.pt
