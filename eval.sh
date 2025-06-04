#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=1:00:00
#SBATCH --output=logs/resent50_surfnorm_eval_test_2.out

python eval_regression_surfnorm_egobody.py --checkpoint /home/kotaik/DH/EgoDepth-HMR/tmp/44899/best_global_model.pt --dataset_root /work/courses/digital_human/13/egobody_release
