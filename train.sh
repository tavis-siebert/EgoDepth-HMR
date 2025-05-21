#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=2:00:00
#SBATCH --output=logs/resent50_depth.out

python train_prohmr_depth_egobody.py --load_pretrained true --checkpoint /work/courses/digital_human/13/weiwan/best_global_model.pt --log_step 250 --num_epoch 5
