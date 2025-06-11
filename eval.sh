#!/bin/bash
#SBATCH --gpus=1
#SBATCH --account digital_human
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_test5.out

python eval_regression_fusion_egobody.py --checkpoint tmp/8893/best_global_model.pt --test_time_optimization
# python eval_regression_fusion_egobody.py --checkpoint tmp/8893/best_global_model.pt
