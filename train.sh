#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=1:00:00

python train_prohmr_depth_egobody.py --data_source real --train_dataset_root egobody_release --val_dataset_root egobody_release

