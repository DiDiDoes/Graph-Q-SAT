#!/bin/bash
#SBATCH --job-name=graphqsat-train-mas-easy-sat
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=mi210:1
#SBATCH -p batch
#SBATCH --array=0-69
#SBATCH --output=logs/graphqsat-train-mas-easy-sat-%A_%a.out
#SBATCH --error=/scratch/%u/graphqsat-train-mas-easy-sat-%A_%a.err

DATASETS=(
  "sr/easy"
  "3-sat/easy"
  "ca/easy"
  "ps/easy"
  "k-clique/easy"
  "k-domset/easy"
  "k-vercov/easy"
)
NUM_SEEDS=5
NUM_DATASETS=${#DATASETS[@]}
TASKS_PER_ALGORITHM=$((NUM_DATASETS * NUM_SEEDS))
TOTAL_TASKS=$((TASKS_PER_ALGORITHM * 2))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
  exit 1
fi

if (( TASK_ID < TASKS_PER_ALGORITHM )); then
  ALGORITHM="dqn"
  ALGORITHM_TASK_ID=${TASK_ID}
else
  ALGORITHM="grpo"
  ALGORITHM_TASK_ID=$((TASK_ID - TASKS_PER_ALGORITHM))
fi

DATASET_INDEX=$((ALGORITHM_TASK_ID / NUM_SEEDS))
SEED=$((ALGORITHM_TASK_ID % NUM_SEEDS))
DATASET=${DATASETS[DATASET_INDEX]}
DATASET_TAG=${DATASET//\//_}

python main.py train \
  --algorithm "${ALGORITHM}" \
  --dataset "${DATASET}" \
  --sat true \
  --seed "${SEED}" \
  --checkpoint-path "checkpoints/graphqsat_${ALGORITHM}_${DATASET_TAG}_seed${SEED}.pth"
