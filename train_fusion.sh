#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=8:00:00
#SBATCH --output=logs/resent50_fusion_concat_trainDepth.out

python train_prohmr_fusion_egobody.py --load_depth_pretrained true --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --log_step 183 --load_pretrained true --checkpoint /work/courses/digital_human/13/weiwan/surf_normals/best_global_model_surf.pt --load_only_backbone true
# python train_prohmr_fusion_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/66270/best_global_model.pt
