#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=3:00:00
#SBATCH --output=logs/preprocess.out

python sam_preprocess.py