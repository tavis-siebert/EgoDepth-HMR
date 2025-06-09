#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account 3dv
#SBATCH --time=12:00:00
#SBATCH --output=logs/resent50_fusion_concat_translloss.out

python train_prohmr_fusion_egobody.py 
    --data_root /work/courses/digital_human/13 \
    --load_depth_pretrained true \
    --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt \
    --log_step 366 \
    --load_pretrained true \
    --checkpoint /work/courses/digital_human/13/weiwan/surf_normals/best_global_model_surf.pt \
    --load_only_backbone true
# python train_prohmr_fusion_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/53289/best_global_model.pt
# python train_prohmr_fusion_flow_egobody.py --load_depth_pretrained true --depth_checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --log_step 366 --load_rgb_pretrained true --rgb_checkpoint /work/courses/digital_human/13/weiwan/surf_normals/best_global_model_surf.pt --log_step 366