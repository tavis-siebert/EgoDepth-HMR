#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human_jobs
#SBATCH --time=8:00:00
#SBATCH --output=logs/eval.out

python eval_regression_fusion_egobody.py --checkpoint tmp/45098/best_global_model.pt --test_time_optimization