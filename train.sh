#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=2:00:00
#SBATCH --output=logs/resent18_fromscratch.out

python train_prohmr_egobody_hha_smplx.py
