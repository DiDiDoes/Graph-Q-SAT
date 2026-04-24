#!/bin/bash
#SBATCH --job-name=graphqsat-train-sat50
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=mi210:1
#SBATCH -p batch
#SBATCH --array=1-5
#SBATCH --output=logs/graphqsat-train-sat50-%A_%a.out
#SBATCH --error=/scratch/%u/graphqsat-train-sat50-%A_%a.err

SEED=$((SLURM_ARRAY_TASK_ID - 1))

python main.py train \
  --dataset 50 \
  --sat true \
  --seed $SEED \
  --checkpoint-path checkpoints/graphqsat/sat50_seed${SEED}.pth
