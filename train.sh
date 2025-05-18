#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=2:00:00
#SBATCH --output=logs/resent50_surfnorm_continue2.out

python train_prohmr_surfnormals_egobody.py --load_pretrained true --checkpoint /home/weiwan/DigitalHuman/EgoDepth-HMR/tmp/53672/best_global_model.pt --log_step 100 
