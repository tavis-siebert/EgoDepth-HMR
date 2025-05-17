#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=2:00:00
#SBATCH --output=logs/resent50_surfnorm.out

python train_prohmr_surfnormals_egobody.py
