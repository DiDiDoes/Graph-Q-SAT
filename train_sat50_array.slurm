#!/bin/bash
#SBATCH --job-name=graphqsat-train-sat50
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=mi210:1
#SBATCH -p batch
#SBATCH --array=1-10
#SBATCH --output=logs/graphqsat-train-sat50-%A_%a.out
#SBATCH --error=/scratch/%u/graphqsat-train-sat50-%A_%a.err

NUM_SEEDS=5
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_TAG="sat50"

if (( TASK_ID < NUM_SEEDS )); then
  ALGORITHM="dqn"
  SEED=${TASK_ID}
  CHECKPOINT_PATH="checkpoints/graphqsat_${DATASET_TAG}_seed${SEED}.pth"
  EXTRA_ARGS=()
else
  ALGORITHM="grpo"
  SEED=$((TASK_ID - NUM_SEEDS))
  CHECKPOINT_PATH="checkpoints/graphqsat_${ALGORITHM}_${DATASET_TAG}_seed${SEED}.pth"
  EXTRA_ARGS=(
    --grpo-buffer-size 64
    --grpo-step-batch-size 64
  )
fi

python main.py train \
  --algorithm "${ALGORITHM}" \
  --dataset 50 \
  --sat true \
  --seed "${SEED}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  "${EXTRA_ARGS[@]}"
