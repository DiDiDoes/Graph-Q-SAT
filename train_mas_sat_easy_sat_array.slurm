#!/bin/bash
#SBATCH --job-name=graphqsat-train-mas-easy-sat
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus=mi210:1
#SBATCH -p batch
#SBATCH --array=0-34
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
TOTAL_TASKS=$((NUM_DATASETS * NUM_SEEDS))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..$((TOTAL_TASKS - 1))" >&2
  exit 1
fi

DATASET_INDEX=$((TASK_ID / NUM_SEEDS))
SEED=$((TASK_ID % NUM_SEEDS))
DATASET=${DATASETS[DATASET_INDEX]}
DATASET_TAG=${DATASET//\//_}

python main.py train \
  --dataset "${DATASET}" \
  --sat true \
  --seed "${SEED}" \
  --checkpoint-path "checkpoints/graphqsat_${DATASET_TAG}_seed${SEED}.pth"
