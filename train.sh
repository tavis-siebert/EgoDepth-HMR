#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=1:00:00

python train_prohmr_egobody_hha_smplx.py
